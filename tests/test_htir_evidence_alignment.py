"""Check tool-result association and the common HTIR evidence projection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from failure_analysis.htir import (
    compile_appworld_htir,
    compile_gaia_htir,
    compile_openhands_htir,
    compile_swe_htir,
    compile_terminal_bench_htir,
)
from failure_analysis.openhands_trace import pair_openhands_tool_events
from failure_analysis.print_htir_compact import build_compact_bundle
from failure_analysis.print_openhands_trace import render_trace, select_evidence


def _write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value))
    return str(path)


def _action(event_id: str, tool_id: str, response_id: str) -> dict:
    return {
        'kind': 'ActionEvent', 'id': event_id, 'tool_call_id': tool_id,
        'tool_name': 'terminal', 'llm_response_id': response_id,
        'action': {'kind': 'TerminalAction', 'command': 'pwd'},
    }


def _result(action_id: str | None, tool_id: str | None, label: str, **observation: object) -> dict:
    return {
        'kind': 'ObservationEvent', 'id': f'result-{label}',
        'action_id': action_id, 'tool_call_id': tool_id, 'tool_name': 'terminal',
        'observation': {'kind': 'TerminalObservation', 'content': label, **observation},
    }


def _compile_oh(tmp_path: Path, events: list[dict], subagents: dict | None = None,
                model_calls: list[dict] | None = None) -> dict:
    trace = _write(tmp_path / 'trace.json', {
        'events': events, 'subagents': subagents or {}, 'model_calls': model_calls or [],
    })
    manifest = _write(tmp_path / 'manifest.json', {
        'trace_path': trace, 'eval_result': {'score': 0, 'feedback': 'Not solved'},
        'resolved': False, 'evaluator_status': 'scored',
    })
    return compile_openhands_htir('task', 'unresolved', {'manifest_path': manifest}, 'Inspect location')


def _tool_result_links(bundle: dict) -> dict[str, dict]:
    nodes = {node['node_id']: node for node in bundle['graph']['nodes']}
    return {
        nodes[edge['to']]['attributes']['observation']['content']: nodes[edge['from']]
        for edge in bundle['graph']['edges']
        if edge['relation'] == 'tool-invocation'
        and nodes[edge['from']]['type'] == 'ToolCallEvent'
        and nodes[edge['to']]['type'] == 'ToolResultRecord'
    }


def _benchmark_bundles(tmp_path: Path) -> list[dict]:
    request = [{'role': 'system', 'content': 'Solve task'}, {'role': 'user', 'content': 'Inspect location'}]
    response = {'role': 'assistant', 'content': 'Inspect', 'extra': {'actions': [{'command': 'pwd'}]}}
    messages = request + [response, {'role': 'user', 'content': 'location', 'extra': {'returncode': 0}}]
    call = {'call_index': 1, 'request_messages': request, 'response_message': response}
    swe_path = _write(tmp_path / 'swe.json', {'messages': messages, 'model_calls': [call], 'info': {}})
    gaia_path = _write(tmp_path / 'gaia.json', {
        'trajectory': [{'tool_calls': [{'name': 'terminal'}], 'observations': 'location'}],
        'model_calls': [call | {'component': 'manager'}], 'info': {'exit_status': 'unsolved'},
    })
    terminal_path = _write(tmp_path / 'terminal.json', {'model_calls': [call], 'info': {'reward': 0}})
    atif_path = _write(tmp_path / 'atif.json', {'steps': [{
        'source': 'agent', 'message': 'Inspect', 'tool_calls': [{'tool_name': 'terminal'}],
        'observation': {'results': [{'content': 'location'}]},
    }]})
    app_response = {'role': 'assistant', 'content': '```python\nprint("location")\n```'}
    app_path = _write(tmp_path / 'appworld.json', {'model_calls': [
        {'call_index': 1, 'request_messages': request, 'response_message': app_response},
        {'call_index': 2, 'request_messages': request + [app_response, {'role': 'user', 'content': 'location'}],
         'response_message': {'role': 'assistant', 'content': 'Done'}},
    ], 'info': {}})
    return [
        compile_swe_htir('task', 'unresolved', {'traj_path': swe_path}, 'Inspect location'),
        compile_gaia_htir('task', 'unresolved', gaia_path, 'Inspect location', 'expected', 'prediction'),
        compile_terminal_bench_htir('task', 'unresolved', {'traj_path': terminal_path, 'atif_path': atif_path}),
        compile_appworld_htir('task', 'unresolved', {'traj_path': app_path}, 'Inspect location'),
        _compile_oh(tmp_path, [_action('action-a', 'call-a', 'response'),
                               _result('action-a', 'call-a', 'location', exit_code=0)]),
    ]


class HtirEvidenceAlignmentTests(unittest.TestCase):

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_reversed_results_share_model_step_and_keep_separate_nodes(self) -> None:
        tmp_path = self.root
        events = [
            _action('action-a', 'call-a', 'response'), _action('action-b', 'call-b', 'response'),
            _result('action-b', 'call-b', 'B'), _result('action-a', 'call-a', 'A'),
        ]
        bundle = _compile_oh(tmp_path, events)
        links = _tool_result_links(bundle)
        assert links['A']['attributes']['source_event_id'] == 'action-a'
        assert links['B']['attributes']['source_event_id'] == 'action-b'
        assert bundle['stats']['model_call_count'] == 1
        assert bundle['outcome_anchors'][0]['attributes']['score'] == 0
        model = next(n for n in bundle['graph']['nodes'] if n['type'] == 'ModelInvocationEvent')
        for node in bundle['graph']['nodes']:
            if node['type'] in {'ParserEvent', 'ToolCallEvent', 'ToolResultRecord'}:
                assert node['step_id'] == model['step_id']
        step = next(s for s in bundle['agent_trace_steps'] if s['numeric_step_id'] == model['step_id'])
        assert len(step['legacy_event_ids']) == 7
        assert len(set(step['legacy_event_ids'])) == 7
        (selected, calls) = select_evidence({'events': events}, [2], [])
        assert [index for (index, _) in selected] == [0, 1, 2, 3]
        assert [call['response_id'] for call in calls] == ['response']

    def test_action_ids_disambiguate_reused_provider_ids_for_compiler_and_reader(self) -> None:
        tmp_path = self.root
        events = [
            _action('action-a', 'reused', 'response-a'), _action('action-b', 'reused', 'response-b'),
            _result('action-a', 'reused', 'A'), _result('action-b', 'reused', 'B'),
        ]
        bundle = _compile_oh(tmp_path, events)
        links = _tool_result_links(bundle)
        assert links['A']['attributes']['source_event_id'] == 'action-a'
        assert links['B']['attributes']['source_event_id'] == 'action-b'
        (selected, calls) = select_evidence({'events': events}, [2], [])
        assert [index for (index, _) in selected] == [0, 2]
        assert [call['response_id'] for call in calls] == ['response-a']
        assert 'result-B' not in render_trace({'events': events}, event_indices=[2])

    def test_uncertain_results_stay_unpaired_and_visible(self) -> None:
        cases = [
            ('reused_tool_id', 'ambiguous', 'duplicate_tool_call_id'),
            ('reused_action_id', 'ambiguous', 'duplicate_action_id'),
            ('conflicting_tool_id', 'conflict', 'tool_call_id_mismatch'),
            ('conflicting_tool_name', 'conflict', 'tool_name_mismatch'),
            ('unknown_action_id', 'unmatched', 'action_id_not_found'),
            ('no_ids', 'unmatched', 'missing_ids'),
            ('before_action', 'conflict', 'result_precedes_action'),
        ]
        for case, status, reason in cases:
            with self.subTest(case=case, status=status, reason=reason):
                tmp_path = self.root
                a = _action('action-a', 'call-a', 'response-a')
                b = _action('action-b', 'call-b', 'response-b')
                result = _result('action-a', 'call-a', 'uncertain')
                events = [a, b, result]
                if case == 'reused_tool_id':
                    b['tool_call_id'] = 'call-a'
                    result['action_id'] = None
                elif case == 'reused_action_id':
                    b['id'] = 'action-a'
                elif case == 'conflicting_tool_id':
                    result['tool_call_id'] = 'call-b'
                elif case == 'conflicting_tool_name':
                    result['tool_name'] = 'different-tool'
                elif case == 'unknown_action_id':
                    result['action_id'] = 'unknown'
                elif case == 'no_ids':
                    result['action_id'] = result['tool_call_id'] = None
                else:
                    events = [result, a, b]
                result_index = events.index(result)
                pair = pair_openhands_tool_events(events)[result_index]
                assert (pair['status'], pair['reason']) == (status, reason)
                bundle = _compile_oh(tmp_path, events)
                assert _tool_result_links(bundle) == {}
                result_node = next(n for n in bundle['graph']['nodes'] if n['type'] == 'ToolResultRecord')
                assert result_node['attributes']['association'] == pair
                assert all(n['step_id'] != result_node['step_id'] for n in bundle['graph']['nodes']
                           if n['type'] == 'ModelInvocationEvent')
                (selected, calls) = select_evidence({'events': events}, [result_index], [])
                assert [index for (index, _) in selected] == [result_index]
                assert calls == []
                assert f'association={status}:{reason}' in render_trace({'events': events})

    def test_partial_legacy_ids_keep_supported_association(self) -> None:
        for (action_id, tool_id, matched_by) in [('action-a', None, 'action_id'), (None, 'call-a', 'tool_call_id')]:
            with self.subTest(action_id=action_id, tool_id=tool_id, matched_by=matched_by):
                tmp_path = self.root
                events = [_action('action-a', 'call-a', 'response'), _result(action_id, tool_id, 'A')]
                bundle = _compile_oh(tmp_path, events)
                assert _tool_result_links(bundle)['A']['attributes']['source_event_id'] == 'action-a'
                result = next((n for n in bundle['graph']['nodes'] if n['type'] == 'ToolResultRecord'))
                assert result['attributes']['association']['matched_by'] == matched_by

    def test_subagent_ids_and_parse_errors_remain_scoped(self) -> None:
        tmp_path = self.root
        main = [_action('same-id', 'same-call', 'main-response'), _result('same-id', 'same-call', 'main')]
        child = [_action('same-id', 'same-call', 'child-response'), _result('same-id', 'same-call', 'child')]
        bad = _action('invalid', 'bad-call', 'error-response')
        bad['action'] = None
        main.extend([bad, {'kind': 'AgentErrorEvent', 'tool_call_id': 'bad-call', 'error': 'Missing required argument'}])
        bundle = _compile_oh(tmp_path, main, {'worker': {'events': child}})
        links = _tool_result_links(bundle)
        assert links['main']['source_ref'] == 'trace.events[0].action'
        assert links['child']['source_ref'] == "trace.subagents['worker'].events[0].action"
        assert links['main']['step_id'] != links['child']['step_id']
        nodes = {node['node_id']: node for node in bundle['graph']['nodes']}
        for node in nodes.values():
            if node['type'] == 'ToolResultRecord':
                assert node['step_id'] == links[node['attributes']['observation']['content']]['step_id']
        assert any(
            edge['relation'] == 'causal' and nodes[edge['from']]['type'] == 'ParserEvent'
            and nodes[edge['to']]['type'] == 'ExceptionEvent'
            and nodes[edge['from']]['step_id'] == nodes[edge['to']]['step_id']
            for edge in bundle['graph']['edges']
        )

    def test_late_results_do_not_rewind_step_allocation(self) -> None:
        events = [
            _action('action-a', 'call-a', 'response-a'),
            _action('action-b', 'call-b', 'response-b'),
            _result('action-a', 'call-a', 'A'),
            _action('action-c', 'call-c', 'response-c'),
            _result('action-b', 'call-b', 'B'), _result('action-c', 'call-c', 'C'),
        ]
        bundle = _compile_oh(self.root, events)
        links = _tool_result_links(bundle)
        assert links['A']['step_id'] < links['B']['step_id'] < links['C']['step_id']
        for node in bundle['graph']['nodes']:
            if node['type'] == 'ToolResultRecord':
                assert node['step_id'] == links[node['attributes']['observation']['content']]['step_id']

    def test_auxiliary_calls_metadata_and_anonymous_actions_keep_distinct_steps(self) -> None:
        events = [
            {'kind': 'LLMCompletionLogEvent'},
            _action('action-a', 'call-a', 'response'), _result('action-a', 'call-a', 'A'),
            {'kind': 'ConversationStateUpdateEvent'},
        ]
        calls = [{'response_id': response_id, 'request_messages': [],
                  'response_message': {'role': 'assistant', 'content': response_id}}
                 for response_id in ('aux-before', 'response', 'aux-after')]
        bundle = _compile_oh(self.root, events, model_calls=calls)
        model_steps = {n['attributes']['response_id']: n['step_id'] for n in bundle['graph']['nodes']
                       if n['type'] == 'ModelInvocationEvent'}
        assert model_steps == {'aux-before': 1, 'response': 2, 'aux-after': 3}
        assert _tool_result_links(bundle)['A']['step_id'] == model_steps['response']
        assert bundle['stats']['model_call_count'] == 3

        actions = [_action('action-a', 'call-a', 'unused'), _action('action-b', 'call-b', 'unused')]
        for action in actions:
            action.pop('llm_response_id')
        anonymous = _compile_oh(self.root, actions + [_result('action-a', 'call-a', 'A'),
                                                     _result('action-b', 'call-b', 'B')])
        links = _tool_result_links(anonymous)
        assert links['A']['step_id'] != links['B']['step_id']

    def test_grouped_multi_tool_results_remain_visible_in_compact_facets(self) -> None:
        actions = [_action(f'action-{i}', f'call-{i}', 'response') for i in range(6)]
        results = [_result(f'action-{i}', f'call-{i}', f'result-{i}', exit_code=0) for i in reversed(range(6))]
        bundle = _compile_oh(self.root, actions + results)
        links = _tool_result_links(bundle)
        step_ids = {n['step_id'] for n in links.values()}
        assert len(step_ids) == 1
        compact = build_compact_bundle(bundle, 100)
        facet = compact['node_facets'][f"step_{next(iter(step_ids)):03d}"]
        result_refs = [r for r in facet['artifact_state']['evidence_refs'] if r['type'] == 'ToolResultRecord']
        assert len(result_refs) == 6
        assert {r['summary'] for r in result_refs} == {f'result-{i}' for i in range(6)}
        assert all(r['association']['status'] == 'matched' for r in result_refs)

    def test_finish_result_shares_submission_step_and_evaluator_stays_separate(self) -> None:
        finish = _action('finish-action', 'finish-call', 'finish-response')
        finish['tool_name'] = 'finish'
        finish['action'] = {'kind': 'FinishAction', 'message': 'Done'}
        result = _result('finish-action', 'finish-call', 'Done')
        result['tool_name'] = 'finish'
        result['observation']['kind'] = 'FinishObservation'
        bundle = _compile_oh(self.root, [finish, result])
        nodes = bundle['graph']['nodes']
        submission = next(n for n in nodes if n['type'] == 'SubmissionEvent')
        observation = next(n for n in nodes if n['type'] == 'ToolResultRecord')
        verification = next(n for n in nodes if n['type'] == 'VerificationEvent')
        assert submission['step_id'] == observation['step_id']
        assert verification['step_id'] > submission['step_id']
        assert [s['numeric_step_id'] for s in bundle['agent_trace_steps']] == [0, 1, 2]
        assert bundle['outcome_anchors'][0]['attributes']['score'] == 0

    def test_result_codes_reach_common_facets_without_reclassifying_execution(self) -> None:
        for (exit_code, timeout) in [(0, False), (1, False), (-1, True), (None, False)]:
            with self.subTest(exit_code=exit_code, timeout=timeout):
                tmp_path = self.root
                result = _result('action-a', 'call-a', 'Program output',
                                 exit_code=exit_code, timeout=timeout, is_error=False)
                bundle = _compile_oh(tmp_path, [_action('action-a', 'call-a', 'response'), result])
                node = next((n for n in bundle['graph']['nodes'] if n['type'] == 'ToolResultRecord'))
                step = next((s for s in bundle['agent_trace_steps'] if s['numeric_step_id'] == node['step_id']))
                refs = bundle['node_facets'][step['step_id']]['artifact_state']['evidence_refs']
                ref = next(r for r in refs if r['type'] == 'ToolResultRecord')
                assert node['attributes']['observation'] == result['observation']
                assert node['attributes']['returncode'] == ref['returncode'] == exit_code
                assert ref['timeout'] is timeout
                assert ref['is_error'] is False
                assert ref['action_id'] == 'action-a'
                assert step['execution_status'] == 'observed'

    def test_mcp_success_does_not_invent_process_exit_code(self) -> None:
        tmp_path = self.root
        result = _result('action-a', 'call-a', 'MCP result', is_error=False)
        result['observation']['kind'] = 'MCPToolObservation'
        bundle = _compile_oh(tmp_path, [_action('action-a', 'call-a', 'response'), result])
        node = next((n for n in bundle['graph']['nodes'] if n['type'] == 'ToolResultRecord'))
        assert node['attributes']['returncode'] is None
        assert node['attributes']['timeout'] is None

    def test_common_reader_preserves_generated_step_messages_and_graph_for_all_compilers(self) -> None:
        tmp_path = self.root
        bundles = _benchmark_bundles(tmp_path)
        common_keys = set(bundles[0])
        common_step_keys = set(bundles[0]['agent_trace_steps'][0])
        for bundle in bundles:
            assert set(bundle) == common_keys
            assert all((set(step) == common_step_keys for step in bundle['agent_trace_steps']))
            compact = build_compact_bundle(bundle, 100)
            full_steps = {step['step_id']: step for step in bundle['agent_trace_steps']}
            for step in compact['agent_trace_steps']:
                for field in ('request_message', 'response_message'):
                    if full_steps[step['step_id']][field]:
                        assert step[field] == full_steps[step['step_id']][field]
            graph = compact['base_graph']
            assert graph['step_node_count'] == len(bundle['base_graph']['nodes'])
            assert graph['edge_count'] == len(bundle['base_graph']['edges'])
            assert graph['edges']
            assert all(edge['from'] in graph['nodes'] and edge['to'] in graph['nodes']
                       for edge in graph['edges'])
            if bundle['mode'] in {'swe', 'openhands'}:
                assert any(ref['returncode'] == 0 for facet in compact['node_facets'].values()
                           for ref in facet['artifact_state']['evidence_refs'] if ref['type'] == 'ToolResultRecord')

    def test_compact_graph_is_bounded_and_retains_boundary_references(self) -> None:
        steps = [{'step_id': f'step_{i:03d}', 'request_message': 'request', 'response_message': 'response'} for i in range(50)]
        bundle = {'agent_trace_steps': steps, 'base_graph': {
            'nodes': [s['step_id'] for s in steps],
            'edges': [{'from': 'step_000', 'to': s['step_id'], 'family': 'temporal'} for s in steps[1:]],
        }}
        compact = build_compact_bundle(bundle, 1)
        assert len([s for s in compact['agent_trace_steps'] if s['step_id'] != '...']) == 1
        graph = compact['base_graph']
        assert len(graph['edges']) == 16
        assert graph['edge_count'] == 49
        assert graph['omitted_edge_count'] == 33
        assert graph['boundary_step_ids'] == [f'step_{i:03d}' for i in range(1, 17)]
        assert graph['step_node_count'] == 50
        with self.assertRaisesRegex(ValueError, 'positive'):
            build_compact_bundle(bundle, 0)

    def test_reader_keeps_legacy_summary_fields_and_count(self) -> None:
        compact = build_compact_bundle({
            'agent_trace_steps': [{'step_id': 's1', 'request_summary': 'old request', 'response_summary': 'old response'}],
            'base_graph': {'step_node_count': 7},
        }, 1)
        assert compact['agent_trace_steps'][0]['request_summary'] == 'old request'
        assert compact['agent_trace_steps'][0]['response_summary'] == 'old response'
        assert compact['base_graph']['step_node_count'] == 7


if __name__ == '__main__':
    unittest.main()
