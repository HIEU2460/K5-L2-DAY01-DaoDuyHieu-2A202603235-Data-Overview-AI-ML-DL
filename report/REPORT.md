# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
    {'sample_id': 'traffic',
    'class_id': 468,
    'class_name': 'cab',
    'rank': 1,
    'score': 0.7825,
    'taxonomy_name': 'ImageNet-1K'}

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Trả lời: Record này nói về cả bức ảnh, vì mô hình classification dự đoán lớp cho toàn bộ ảnh đầu vào, không tạo bounding box để xác định một vật thể riêng biệt.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Trả lời: Danh sách tên lớp có sẵn trong taxonomy đi kèm checkpoint đã huấn luyện. Model không tự nghĩ ra tên lớp mới khi dự đoán.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Trả lời:  Cần ghi đủ ba trường để xác định rõ nhãn dự đoán. class_id là mã lớp, class_name là tên dễ đọc, còn taxonomy_name cho biết bộ danh sách lớp được sử dụng, tránh nhầm lẫn giữa các taxonomy khác nhau.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Trả lời: Guideline cần quy định rõ cách xác định chủ thể chính khi ảnh có nhiều vật thể, tiêu chí ưu tiên lựa chọn lớp và cách xử lý trường hợp có nhiều chủ thể quan trọng như nhau.

6. Vì sao không được lấy `score` của model làm ground truth?
Trả lời: Score chỉ thể hiện mức độ tin cậy của model đối với prediction, không đảm bảo dự đoán đúng. Ground truth phải được gán hoặc xác nhận theo guideline, không thể lấy trực tiếp từ score của model.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
Trả lời: class_name = person; score = 0.912625; bbox_xyxy = [385.33, 69.24, 498.92, 348.92]; bbox_width = 113.58; bbox_height = 279.68.

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Trả lời: 
- Góc trên bên trái: (385.33, 69.24).
- Góc dưới bên phải: (498.92, 348.92).
- Chiều rộng: 113.58 pixel.
- Chiều cao: 279.68 pixel.
 Các tọa độ được tính từ góc trên bên trái của ảnh, với trục X tăng từ trái sang phải và trục Y tăng từ trên xuống dưới. Bounding box là hình chữ nhật bao quanh đối tượng person được mô hình phát hiện.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Trả lời: Tôi chọn hai ngưỡng confidence là 0.20 và 0.60.
- Ở ngưỡng 0.20: mô hình phát hiện 17 prediction.
- Ở ngưỡng 0.60: mô hình phát hiện 6 prediction.
=> Khi giảm ngưỡng confidence từ 0.60 xuống 0.20, số prediction tăng từ 6 lên 17, tức tăng thêm 11 prediction. Điều này xảy ra vì ngưỡng thấp cho phép giữ lại nhiều dự đoán có score thấp hơn, nhưng cũng có thể làm tăng số lượng dự đoán sai cần kiểm tra.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Trả lời: Khi hạ ngưỡng confidence từ 0.60 xuống 0.20, số prediction tăng từ 6 lên 17. Reviewer phải kiểm tra thêm 11 prediction, đặc biệt là các dự đoán có score thấp, để phát hiện và loại bỏ các vật thể nhận diện sai (false positive), kiểm tra nhãn lớp (class_name), vị trí và kích thước bounding box, đồng thời xử lý các box trùng lặp hoặc không chính xác. Những prediction không đạt yêu cầu cần được sửa hoặc đưa về bước rework.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Trả lời: Khi gán nhãn vật thể, annotator phải vẽ bounding box hình chữ nhật bao sát phần vật thể nhìn thấy được, không lấy thừa vùng nền và không bỏ sót các phần có thể quan sát được. Mỗi vật thể cần được gán đúng nhãn lớp (class_name), đồng thời áp dụng quy tắc này thống nhất cho tất cả các ảnh trong bộ dữ liệu.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trả lời: Trong ảnh kitchen, đối tượng oven ở bên trái bị cắt bởi mép ảnh. Tôi đề xuất quy định trong guideline rằng khi vật thể bị cắt mép, annotator chỉ vẽ bounding box bao quanh phần vật thể nhìn thấy được, không tự suy đoán phần nằm ngoài ảnh. Nếu không xác định được ranh giới hoặc loại vật thể, cần hỏi Lab Coach để được hướng dẫn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
Trả lời: 
    Số instance của kitchen: 11
    instance_id = kitchen-001
    class_name = person
    score = 0.899318
    polygon_point_count = 348
    5 điểm đầu polygon_xy = [[446.0, 70.0], [445.0, 71
    .0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Trả lời: So với bounding box chỉ biểu diễn vật thể bằng một hình chữ nhật, polygon sử dụng nhiều điểm tọa độ để mô tả đường viền thực tế của vật thể. Nhờ đó, polygon thể hiện hình dạng vật thể chính xác hơn, đặc biệt với những vật thể có đường biên cong hoặc không đều, đồng thời hạn chế lấy thừa vùng nền.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
Trả lời: instance_id là mã định danh riêng cho từng đối tượng được phát hiện trong một ảnh, giúp phân biệt nhiều đối tượng thuộc cùng một lớp. Nó không phải class_id, vì class ID xác định loại vật thể, ví dụ person hoặc bowl. Nó cũng không phải tracking ID, vì không dùng để theo dõi cùng một vật thể qua nhiều khung hình video.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Trả lời:  Khi gán nhãn instance segmentation, annotator phải vẽ đường biên mask bám sát ranh giới phần vật thể nhìn thấy được, không lấy thừa vùng nền và không tự suy đoán phần vật thể bị che khuất. Mỗi vật thể riêng biệt phải có mask riêng, kể cả khi chúng thuộc cùng một lớp.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trả lời: Trong ảnh kitchen, ở góc dưới bên trái có nhiều chiếc bát (bowl) đặt gần nhau, khiến việc xác định chính xác đường biên mask của từng chiếc bát gặp khó khăn. Tôi sẽ sử dụng guideline để vẽ mask riêng cho từng chiếc bát, chỉ bao quanh phần vật thể nhìn thấy được và không gộp nhiều chiếc bát thành một mask. Nếu ranh giới giữa các vật thể không thể xác định rõ, tôi sẽ escalation cho Lab Coach để được hướng dẫn, tránh tự suy đoán gây sai lệch dữ liệu.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

Trả lời:

| Tác vụ | Ground truth (đơn vị nhãn) | Lỗi hoặc chỗ khó trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| ------ | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp (class_id, class_name) cho toàn ảnh. | Ảnh traffic có nhiều đối tượng, khó chọn một lớp đại diện. | Chọn nhãn toàn ảnh theo taxonomy và guideline. | Kiểm tra nhãn đúng nội dung ảnh, đúng taxonomy. |
| Phát hiện vật thể | Mỗi vật thể có nhãn lớp và box [x1, y1, x2, y2]. | Ảnh kitchen: ngưỡng 0.20 có 17 prediction, ngưỡng 0.60 có 6; box oven bên trái bị cắt mép ảnh. | Gán đúng lớp, vẽ box sát phần vật thể nhìn thấy. | Kiểm tra box lệch, sai lớp, trùng box và bỏ sót vật thể. |
| Instance segmentation | Mỗi vật thể có nhãn lớp và mask/polygon riêng (polygon_xy). | Ảnh kitchen: các bowl ở góc dưới trái nằm gần nhau, khó tách biên mask. | Vẽ mask sát biên, tách riêng từng vật thể. | Kiểm tra mask lấn nền, sai biên và gộp nhầm các instance. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Trả lời: Chỉ sử dụng ảnh và dữ liệu được phép trong phạm vi bài lab, không chia sẻ thông tin cá nhân, dữ liệu nhạy cảm hoặc tài liệu nội bộ lên GitHub.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Trả lời: Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi cho phép, tôi sẽ dừng xử lý ngay và báo cho Lab Coach hoặc giảng viên phụ trách để được hướng dẫn, không tự ý sử dụng hoặc công khai dữ liệu.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [v] `REPORT.md` (file này đã điền)
- [v] `day1_lab_outputs/classification_predictions.json`
- [v] `day1_lab_outputs/detection_predictions.json`
- [v] `day1_lab_outputs/segmentation_predictions.json`
- [v] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [v] `day1_lab_outputs/visuals/classification_top5.png`
- [v] `day1_lab_outputs/visuals/detection_predictions.png`
- [v] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [v] Ô validation cuối notebook báo `PASS`
- [v] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [v] Đã nộp **link repository** trên LMS
