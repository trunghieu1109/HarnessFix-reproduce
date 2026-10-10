# Sửa lỗi redaction đường dẫn — 2026-10-09

## Nguyên nhân

`is_credential_name()` chuyển tên biến sang chữ thường trước khi kiểm tra khóa
`pwd`. Vì vậy, biến shell `PWD` bị nhận nhầm là mật khẩu. Giá trị đường dẫn repo
được thêm vào danh sách secret và bị thay bằng `[REDACTED]` trong prompt,
observations và artifacts phục vụ analysis/aggregate.

## Những thay đổi trong đợt sửa này

- `failure_analysis/secret_redaction.py`: loại trừ chính xác `PWD` và `OLDPWD`
  trước khi chuyển tên sang chữ thường. Đường dẫn thư mục được giữ nguyên;
  trường dữ liệu `pwd`, `password`, API key và authorization vẫn được che.
- `tests/test_openhands_prompt_safety.py`: thêm hai kiểm tra hồi quy. Một kiểm tra
  việc giữ đường dẫn/source references trong khi vẫn che credential; một kiểm
  tra request thực tế của agent và thực thi lệnh đọc file từ đường dẫn mà model
  nhận được, đồng thời kiểm tra biến thư mục và trajectory được lưu.
- Thêm ghi chú này để lưu nguyên nhân, phạm vi sửa và kết quả kiểm tra.

Runtime của SLM/Better Harness, model configs, các prompt, chính sách tối ưu/gate,
hai tài liệu `sdk_reference`/`read_trajectory` và artifacts của các run cũ được
giữ nguyên trong đợt sửa này.

## Kiểm tra

Hai kiểm tra mới tái hiện lỗi trước khi sửa. Sau khi sửa, 17 kiểm tra đều đạt:

```bash
cd /root/HarnessFix-reproduce
LITELLM_LOCAL_MODEL_COST_MAP=True OPENHANDS_SUPPRESS_BANNER=1 \
  .venv/bin/python -B -m unittest \
  tests.test_openhands_prompt_safety tests.test_openhands_repair_references
```

Đã kiểm tra thêm các đường dẫn manifest, HTIR, analysis, plan, iteration report
và aggregate context của run Stock Alert
`stock_alert_20261008T182226443474Z_909072`: đường dẫn qua redaction giữ nguyên
và các file đọc được. Không gọi Gemini/Qwen hoặc chạy lại benchmark trong kiểm tra.

## Khi chạy lại

Nội dung đã bị che trong artifacts cũ không tự khôi phục. Chạy analysis lại từ
traces gốc với output mới để tạo evidence/diagnosis theo code đã sửa.
Kiểm tra này chưa xác nhận Gemini sẽ đọc hết context hoặc tạo diagnosis tốt hơn.

`secret_redaction.py` nằm trong fingerprint source của pipeline đóng vòng.
Run cũ vẫn giữ fingerprint cũ; dùng run-dir mới nếu chạy toàn bộ pipeline bằng
code mới. Có thể chạy riêng bước `analyze` trên traces có sẵn để kiểm tra trước,
không cần chạy lại task runtime.
