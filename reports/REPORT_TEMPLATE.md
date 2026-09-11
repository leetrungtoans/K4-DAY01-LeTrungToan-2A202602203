# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
    `class_id`: 468, `class_name`: "cab", `rank`: 1, `score`: 0.510915, `taxonomy_name`: "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
    Model dự đoán toàn bộ ảnh traffic có khả năng thuộc lớp cab cao nhất với score 0.510915
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    Taxonomy ImageNet-1K và checkpoint được huấn luyện trên taxonomy đó định nghĩa các class mà model có thể dự đoán
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    Giữ ID để định danh chính xác class, tên lớp và tên taxonomy giúp con người hiểu được lớp đó là gì và biết class đó thuộc hệ phân loại nào
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    Nếu ảnh có nhiều chủ thể, guideline cần quy định rõ gán nhãn cho từng ảnh hay từng chủ thể và cách xử lý khi có nhiều class cùng xuất hiện
- Vì sao model score không phải ground truth?
    Vì score chỉ là mức độ model dự đoán class đó, còn ground truth phải đến từ nhãn thực tế từ con người

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    `class_name`: "person", `score`: 0.912625, `bbox_xyxy`: [385.33, 69.24, 498.92, 348.92], `bbox_width`: 113.58, `bbox_height`: 279.68
- Diễn giải vị trí box bằng lời:
    Box person nằm ở phía bên phải ảnh, bắt đầu khoảng (385,69) và kết thúc (499,349), bao quanh gần như toàn bộ người đang đứng trong bếp
- So sánh số prediction ở hai threshold:
    Ở threshold 0.35, model tạo ra 11 prediction, trong khi ở threshold 0.50 chỉ còn 6 prediction, nghĩa là khi tăng threshold từ 0.35 lên 0.50 thì 5 prediction có score thấp hơn 0.50 bị loại, chỉ giữ lại những box mà model có mức tin cậy từ 0.50 trở lên
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    Threshold thấp tăng độ bao phủ nhưng tạo nhiều prediction hơn để reviewer kiểm tra, còn threshold cao giảm số box và khối lượng reviewer nhưng có nguy cơ bỏ sót object khó hoặc bị che khuất
- Đề xuất một quy tắc box chặt:
    Box nên ôm sát toàn bộ phần object nhìn thấy, không chứa khoảng nền dư thừa và không cắt vào phần object đang nhìn thấy, đồng thời các tọa độ phải nằm trong phạm vi ảnh
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?đồng thời đánh dấu trường hợp không 
    Guideline cần quy định cách vẽ box cho object bị che khuất hoặc bị cắt bởi biên ảnh, đủ thông tin và escalate cho reviewer khi không thể xác định rõ phạm vi object

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    `instance_id`: "kitchen-001", `class_name`: "person", `score`: 0.899318,
        số điểm và một phần `polygon_xy`: [
        [
            446.0,
            70.0
        ],
        [
            445.0,
            71.0
        ],
        [
            444.0,
            71.0
        ],
        [
            443.0,
            72.0
        ],...]
- Polygon bổ sung chi tiết gì so với box?
    Box chỉ cho biết hình chữ nhật bao quanh object, còn polygon mô tả đường biên thực tế của object theo nhiều điểm, nên có thể thể hiện chính xác hình dạng người, bàn, đồ vật... và loại bỏ phần nền nằm bên trong box
- `instance_id` dùng để làm gì và không phải loại ID nào?
    instance_id dùng để phân biệt từng object cụ thể trong cùng một ảnh, không phải class_id
- Đề xuất một quy tắc biên mask:
    Polygon nên bám sát đường biên phần object nhìn thấy, không ăn sang nền hoặc object khác, nhưng cũng không được bỏ sót phần object có thể xác định rõ
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    Guideline cần quy định cách xử lý khi ranh giới object không rõ, hai object tiếp xúc hoặc một object bị che khuất, và nếu reviewer không thể xác định đáng tin cậy đường biên thì nên đánh dấu cần review/escalation thay vì tự đoán

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn cho toàn ảnh | Nhiều chủ thể, khó chọn nhãn | Gán nhãn theo guideline | Kiểm tra nhãn đúng |
| Phát hiện vật thể | Class + Bounding box | Box rộng/hẹp, che khuất | Vẽ box sát vật thể | Kiểm tra class + box |
| Instance segmentation | Class + Polygon/Mask | Biên mờ, tiếp xúc, che khuất | Vẽ mask sát vật thể | Kiểm tra class + biên mask |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng dữ liệu đúng phạm vi được giao, không tự ý sao chép, chia sẻ hoặc đưa dữ liệu ra ngoài
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Quản lý phụ trách dự án

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
