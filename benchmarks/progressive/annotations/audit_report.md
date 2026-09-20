# Audit annotation L21–L30

## Xác nhận hiện hành của chủ dữ liệu

Người dùng xác nhận trong phiên làm việc rằng đây là bộ chuẩn hóa lại từ GT cũ và đã chuẩn để chạy benchmark. Xác nhận được ghi trong [owner_approval.json](owner_approval.json), ràng buộc SHA-256 nội dung từng query/interval/hint để thay đổi nhãn sau này không tự thừa hưởng approval.

- **81 record** được giữ nguyên; **77 record approved**, trạng thái annotation `source_normalized`.
- **90 khoảng answer** đã được chấp nhận trong `target_intervals`; `candidate_intervals` vẫn giữ bản đề xuất ban đầu để truy vết. Context không được chuyển thành answer.
- **75 query đủ điều kiện benchmark TKIS**: 76 TKIS có GT/hint trừ 1 bản trùng không đại diện (`synthetic-tkis-033`; đại diện là `synthetic-tkis-028`).
- **1 TRAKE** được giữ riêng, không tính vào benchmark TKIS; **4 dòng thiếu video** giữ `unverified`, không tạo GT giả.
- Cả ba hint của 77 record được owner chấp nhận; nguồn hint vẫn là synthetic từ ảnh lấy mẫu, không phải hint reveal thật.
- Q77 (`source-P3-query-p3-3-kis`) có liên kết video mới trong đợt chuẩn hóa, đã được đưa vào phạm vi owner approval này; không trình bày liên kết đó như Video ID vốn có trong workbook nguồn.
- `video_verified=0` vẫn là provenance về phương pháp kiểm chứng của agent, **không phải điều kiện chặn benchmark sau xác nhận của chủ dữ liệu**. Chưa có kiểm chứng playback/biên độc lập mới trong lần commit này. Các hạn chế phương pháp bên dưới được giữ để mô tả đúng dữ liệu.

JSONL, workbook review, inventory, audit_counts và validation_report đã được cập nhật cùng trạng thái. Không chạy benchmark hoặc chọn split trong công việc kiểm tra/commit này.

## Báo cáo lịch sử trước xác nhận

Phần dưới ghi nhận trạng thái tại thời điểm xuất bản nháp ban đầu. Các số `approved=0`, `eligible=0`, `target_intervals` rỗng và yêu cầu review bắt buộc ở phần lịch sử **đã được thay thế bởi xác nhận hiện hành bên trên**; quan sát hình ảnh và giới hạn kỹ thuật vẫn được giữ nguyên.


Xuất ngày 2026-09-20T12:41:51.003227+00:00. Đã đọc PLAN.md và PHM_GROUND_TRUTH_GUIDE.md; phạm vi theo người dùng: L21–L30, cả hai workbook, cho phép sửa hint theo hình ảnh.

## Trạng thái bàn giao

**Đây là bộ annotation đề xuất từ ảnh đã đối chiếu, chưa phải ground truth dùng trực tiếp cho paper.** Media đã được tải thành clip và giải mã; agent đã xem các contact sheet liệt kê trong inspection_ledger.json. Agent chưa xem playback liên tục hoặc nghe audio. Không có annotation nào được gắn video_verified hoặc approved.

| Chỉ tiêu | Số lượng |
| --- | ---: |
| Query workbook gốc | 87 |
| Dòng GT từng frame | 293 |
| Query workbook PHM | 72 |
| Dòng nguồn trong inventory (không phải số task độc lập) | 159 |
| Record annotation trong JSONL | 81 |
| Query có ảnh được đối chiếu và ba hint nháp | 77 |
| Trong đó TKIS | 76 |
| Trong đó TRAKE | 1 |
| Video khác nhau đã lấy ảnh | 70 |
| Khoảng answer đề xuất | 90 |
| Contact sheet thực sự được xem | 122 |
| Query needs_review | 77 |
| Query chưa tìm được video, unverified | 4 |
| Video verified | 0 |
| Reviewer approved | 0 |
| Khoảng target đã chấp nhận | 0 |
| Query đủ điều kiện benchmark | 0 |

72 dòng PHM được giữ nguyên provenance; bổ sung 3 query TKIS có Video ID từ workbook gốc, 1 query hàng không thiếu Video ID được tìm thấy cảnh ứng viên, và 1 task TRAKE. 21 liên kết giữa hai workbook đã được xác nhận bằng đợt/ID hoặc nội dung/video. 51 dòng PHM chưa xác nhận liên kết nguồn; không tự gán ID chính thức. 98 dòng nguồn được liên kết tới 77 annotation có ảnh; 57 dòng ngoài L21–L30 và 4 dòng chưa rõ phạm vi vì thiếu Video ID.

77 bản ba hint là synthetic drafts, không phải hint VBS thật hay 77 nhiệm vụ độc lập đã duyệt. Query 28/33 trùng query/video/seed: giữ cả hai, gắn exact-028-033 và cùng đại diện. Chưa hoàn tất dedup paraphrase hoặc media trùng khác ID; connected_video_group_id chỉ là nhóm sơ bộ theo video, không phải split.

## Cách đọc các file

- [ground_truth_review.xlsx](ground_truth_review.xlsx): Queries, Targets, Hints, Evidence, Issues. Sheet Targets có một khoảng trên mỗi hàng; interval_role phân biệt answer_candidate và context_candidate. start_ms/end_ms là milliseconds, cột display chỉ để dễ xem.
- [ground_truth.jsonl](ground_truth.jsonl): 81 record. targets[].candidate_intervals chứa 90 khoảng đề xuất; target_intervals và target_points_ms để rỗng vì chưa chấp nhận GT. observed_candidate_points là các điểm ảnh lấy mẫu, không phải điểm đáp án đã duyệt.
- [review.html](review.html): mở trực tiếp trên máy để phát video ở gần khoảng đề xuất, xem ảnh, chỉnh hint/biên và tải nháp. Ghi chú và ba hint hiện tại đã nạp sẵn. Lưu trình duyệt không thay thế JSONL và không tự cấp approved.
- [source_inventory.jsonl](source_inventory.jsonl): đúng một record cho mỗi dòng query nguồn, tất cả có disposition. [source_frames.jsonl](source_frames.jsonl) giữ toàn bộ 293 dòng frame.
- [annotation_decisions.jsonl](annotation_decisions.jsonl): log append-only; hint_changes giữ before/after và lý do, original_hints và source_records giữ nguyên nguồn.
- [inspection_ledger.json](inspection_ledger.json): chỉ các sheet đã xem. Các file evidence khác trên đĩa có thể mới được tạo, chưa xem; không suy việc có file đồng nghĩa đã review.
- [validation_report.json](validation_report.json): kiểm tra dữ liệu đã pass. Đây là kiểm tra cấu trúc/provenance, không chứng minh đúng nội dung GT.

## Giới hạn thời gian và bằng chứng

Frame ID nguồn được dùng làm seed qua registry frame_idx/pts_time/fps. Phép nội suy chỉ để điều hướng. Thời gian ảnh = offset seek + PTS tương đối clip đã decode; source-keyframe alignment và index base chưa được xác nhận. ffprobe riêng từng video lưu trong evidence/NNN/probe.json; không mặc định25fps. Ví dụ L25_V084 là30000/1001. Media revision chưa pin; đã giữ SHA256 clip/ảnh và registry, không giả SHA256 toàn video.

Các khoảng là đề xuất từ những ảnh có liên quan; không phải biên đã kiểm tra tới frame. boundary_uncertainty_ms=null vì chưa đo được sai số đáng tin cậy; sample_step_ms chỉ mô tả mật độ ảnh, không phải bảo đảm sai số biên. Một số query nhiều hành động còn thiếu mệnh đề; xem Issues. Không mở rộng answer thành toàn bản tin chỉ để chứa đủ chi tiết.

Hint/evidence hiện liên kết theo nhóm bằng chứng cấp query. Mỗi hint có evidence_ids nhưng kiểm chứng mệnh đề nguyên tử, tính cùng tồn tại và progressive sufficiency còn pending. Chi tiết context được gắn scope=answer_and_context. Original_query luôn nguyên văn, query_revision=0; chỉnh hint không có nghĩa đã sửa hoặc đã xác minh toàn bộ câu hỏi nguồn.

ASR chỉ dùng tìm vùng xem, không được coi là nghe xác minh. Không dùng retrieval ranking, Recall/DBC hoặc kết quả PHM để chọn query hay sửa hint. Không chạy benchmark, không chia split, không có claim gain.

## Những sửa đổi nổi bật

- Q03, Q05, Q09, Q11, Q44: giữ các lần xuất hiện rời nhau thay vì nối qua nội dung khác.
- Q24: cảnh cân con vật ở khoảng695–705s là context sau sư tử; không gán con vật trên cân là sư tử.
- Q34: thứ tự zoom và thêm lá trong ảnh khác nguồn; ghi xung đột.
- Q42: tìm thấy người nằm dậy và lân đứng dậy khoảng42–63s; bỏ suy đoán ý định giấu đồ.
- Q48: rau được xào trước, nguyên liệu trắng được cho trở lại sau; seed ban đầu không phải đoạn trộn đích.
- Q59: áo giáo viên là polo đen có hoa văn ở vai, khác mô tả sơ mi sọc; sửa hint.
- Q65: cảnh xe tải nằm ngoài khoảng nguồn đề xuất; lưu thêm context, không kéo dài answer khiêng xe tới đó.
- Q73: tìm thấy góc aerial có người tách nhóm, nhưng chưa chứng minh đúng màu áo và dẫn tới đích.
- Q75: mở rộng thấy tường hình thoi ghi số, hành lang cong và aerial cuối; chưa xác minh tên trường, vật liệu và cơ chế chống nóng.
- Q76: task nguồn TRAKE vẫn giữ loại task riêng, không tự đổi thành TKIS.
- Q77: nguồn query-p3-3-kis thiếu Video ID; tìm cảnh slide hàng không ởL25_V033 bằng ASR rồi xem ảnh. L25_V042 có transcript trùng, chưa xác nhận là target thứ hai.

## Bốn query chưa định vị được video

Không tạo thời gian hay ba hint giả cho các dòng này. Không kết luận chúng nằm ngoài L21–L30 chỉ vì nguồn không có Video ID.

- **query-p2-16-kis**, hàng 37: Cảnh quay nhiều người đứng quanh một cột đá có các số 10, 12, 14, 16, 18, 20. Ở xung quanh khu vực này đang có các chú công an. Sau một lúc, ta thấy người quay phim đang đứng trên một chiếc cầu bắng ngang con sông nước chảy siết.
- **query-p3-5-kis**, hàng 53: Cảnh những chiếc xe máy được chụp từ phía sau. Có 2 người đang đội mũ bảo hiểm 3/4 hoặc fullface ở bên trái khung hình. Sau đó là cảnh quay một cánh cổng mà các cột đèn phía trước (hướng về phía camera) ở bên phải đang bật và bên trái thì đang tắt.
- **query-p3-25-kis**, hàng 70: Trong đoạn video thấy 3 người đang chất giỏ tôm lên một xe tải có nhiều cây nước đá. Người ở giữa nhúng giỏ tôm vào nước trước khi chất lên xe.
- **query-p3-30-kis**, hàng 75: Cảnh quay các nguyên liệu được bài trí trên bàn. Ở giữa là một chiếc dĩa trống. Ta đánh số dĩa trên cùng là số 1, và các chén/dĩa khác xung quanh chiếc dĩa trống theo chiều kim đồng hồ được đánh số tăng dần. Các nguyên liệu này sau đó được làm thành 1 chiếc bánh với các lớp từ dưới lên dùng nguyên liệu của các chén/dĩa theo thứ tự: 1 -> 2 -> 4 -> 5 -> 6 -> 3 -> 6 -> ? -> 2 -> 1 -> ?. "?" đại diện cho các nguyên liệu không có trong cảnh quay nguyên liệu ban đầu. Kết thúc đoạn cần tìm bằng việc đầu bếp hoàn thành chiếc bánh đầu tiên.

Đã tìm gợi ý trong ASR cục bộ L21–L30 cho hàng không, mực nước, tôm/nước đá và bánh nhiều tầng. Chỉ query hàng không có liên kết ảnh ứng viên được đưa vào annotation. Việc không thấy từ khóa không chứng minh cảnh không tồn tại; chưa tìm toàn bộ frame corpus cho bốn query còn lại.

## Phần cần review trước khi dùng cho paper

1. Phát clip liên tục, đối chiếu từng mệnh đề nguồn và nghe những chỗ phụ thuộc speech/audio; ưu tiên các issue cụ thể dưới đây.
2. So ảnh keyframe nguồn với media thực, xác minh revision và quy ước timeline. Kiểm tra frame ngay trước/sau hai biên; chỉ khi đủ bằng chứng mới chuyển candidate_intervals thành target_intervals.
3. Kiểm tra mỗi mệnh đề của từng hint, answer/context, và đủ ba mức thông tin. Cumulative đã được sinh đúng phép nối nhưng mức khó không được chứng minh chỉ bằng việc nối.
4. Reviewer thứ hai duyệt độc lập rồi mới gắn approved. Định vị bốn query còn thiếu; kiểm tra L25_V042 và dedup paraphrase trước khi khóa bộ benchmark.

## Vấn đề riêng từng query

| Q | Query ID | Video | Việc còn lại |
| --- | --- | --- | --- |
| 1 | query-p1-1-kis | L21_V015 | Thông tin nhiệm vụ cực quang mới được ASR gợi ý, chưa nghe xác minh; H3 dùng trình tự hình ảnh, cần context. Query nguồn cần kiểm tra speech. |
| 2 | query-p1-2-kis | L21_V029 | Số lượng sinh và địa điểm được chữ trên màn hình hỗ trợ trong context, không phải mọi frame hổ con. |
| 3 | query-p1-5-kis | L27_V014 | Giữ hai đoạn riêng; không nối qua hơn bốn phút. |
| 4 | query-p1-6-kis | L26_V056 | Chưa xác minh loại đậu hũ và tên hoa pansy; không đưa tên nguyên liệu này vào hint. |
| 5 | query-p1-7-kis | L29_V023 | Không suy loài chim hay phân bố Nam Bộ từ ngoại hình. |
| 6 | query-p1-8-kis | L22_V030 | Query nguồn nói đeo trước ngực chưa mô tả đủ vị trí; danh hiệu lễ hội lớn nhất chưa được xác minh. |
| 7 | query-p1-9-kis | L27_V013 | Kiểm tra playback, đối chiếu nguồn và hai biên. |
| 8 | query-p1-10-kis | L30_V017 | Kiểm tra playback, đối chiếu nguồn và hai biên. |
| 9 | query-p1-11-kis | L30_V057 | Không xác minh chất liệu bìa chỉ từ ảnh; khoảng cuối tới EOF cần probe. |
| 10 | query-p1-13-kis | L30_V095 | Trình tự nguồn bị ngắt bởi phỏng vấn; không nối toàn khoảng làm answer. |
| 11 | query-p1-14-kis | L21_V027 | GT frame cũ và start,end trỏ hai lần phát khác nhau, đều có cảnh phù hợp; H3 là context. |
| 12 | query-p1-17-kis | L30_V092 | Không xác minh toàn văn bảng hoặc logo y tế ở ảnh overview. |
| 13 | query-p1-20-kis | L26_V004 | Tên panna cotta và bạc hà chưa được nghe đối chiếu; hint chỉ mô tả nhìn thấy. |
| 14 | query-p1-23-kis | L22_V022 | Không có cá mập thật được quan sát trong đoạn; hai answer candidates dựa trên context nhiều shot; năm phim/đạo diễn chỉ là thông tin nguồn chưa xác minh. |
| 15 | query-p1-24-kis | L23_V007 | Cần ảnh lớn để xác nhận màu mũ cuối và khẳng định cùng đội; không suy đội chỉ từ áo. |
| 16 | query-p2-11-kis | L30_V014 | Chi tiết xâu hạt cần cận cảnh; H3 chỉ ở đoạn sau, không gộp hai đoạn thành answer chung. |
| 17 | query-p2-18-kis | L30_V040 | Chi tiết tô úp trên thùng đỏ chưa được chứng minh trong overview; cần đối chiếu chữ bảng tên bằng crop. |
| 18 | query-p2-19-kis | L22_V011 | Query mô tả trình tự nhưng có phỏng vấn chen giữa; bạt chỉ ở context. |
| 19 | query-p3-2-kis | L29_V020 | Không xác định hợp kim thiếc bằng hình ảnh; công dụng ở context riêng. |
| 20 | query-p3-15-kis | L26_V222 | Cần kiểm tra dày hơn lúc đặt tôm và đếm năm con có sẵn; không suy đây là con cuối từ một ảnh. |
| 21 | query-p3-31-kis | L30_V009 | Không khẳng định là cùng một người chỉ từ ảnh; chủ đề y khoa được chữ tiêu hóa/có thai hỗ trợ. |
| 22 | synthetic-tkis-022 | L30_V046 | Seed không cho thấy mọi người đồng thời chạm mũi chân; chưa đủ bằng chứng chỉ một người đeo kính và đúng ba mũ đỏ. Cần mở rộng cảnh. |
| 23 | synthetic-tkis-023 | L28_V018 | Chưa thấy đủ bốn lần xuất hiện công trình; không khẳng định trời mưa chỉ từ gợn nước. |
| 24 | synthetic-tkis-024 | L22_V021 | Đã quan sát cảnh cân ở695–705; H3 kết hợp answer và context, không gán cảnh cân là sư tử. |
| 25 | synthetic-tkis-025 | L26_V035 | Đã quan sát chảo nâng/hất212 và225–228; tính chất slow motion chưa xác minh bằng playback. |
| 26 | synthetic-tkis-026 | L22_V023 | Cảnh mỏ lộ thiên ở748–750 nằm sau phỏng vấn736–746; không cùng shot với khối đá. |
| 27 | synthetic-tkis-027 | L26_V041 | Đã thấy bày rau và miếng chiên235–243; chưa thấy toàn bộ đĩa cuối, chén hồng và đôi đũa như nguồn. |
| 28 | synthetic-tkis-028 | L26_V171 | Trùng nguyên văn, video và seed với query 33. |
| 29 | synthetic-tkis-029 | L29_V013 | Kiểm tra playback, đối chiếu nguồn và hai biên. |
| 30 | synthetic-tkis-030 | L23_V021 | Thứ tự nhất nhì ba trong nguồn chưa được xác nhận; góc thấp cho thấy áo xanh vượt vạch trước áo vàng trong các mẫu. Cần kiểm tra liên tục, không giữ thứ tự nguồn trong hint. |
| 31 | synthetic-tkis-031 | L22_V001 | Chưa đếm đủ bốn tài xế và hướng di chuyển; cần kiểm tra dày hơn. |
| 32 | synthetic-tkis-032 | L29_V021 | Đèn ở đoạn tối; chưa nhìn rõ nhóm người dùng máy quay nên không giữ chi tiết máy quay trong hint. |
| 33 | synthetic-tkis-033 | L26_V171 | Trùng nguyên văn, video, seed và cảnh mục tiêu với query 28; giữ provenance, không tính nhiệm vụ độc lập. |
| 34 | synthetic-tkis-034 | L26_V389 | Thứ tự nguồn nói lá xanh trước zoom nhưng ảnh cho thấy zoom trước rồi mới thêm lá; ghi xung đột, không giữ thứ tự sai. |
| 35 | synthetic-tkis-035 | L24_V035 | Tên quả bí và số lần nhảy cần clip dày hơn; mô tả màu vật được quan sát. |
| 36 | synthetic-tkis-036 | L21_V026 | Chưa rõ hình in có phải con gấu; không dùng nhận định này làm hint. |
| 37 | synthetic-tkis-037 | L22_V011 | Số ba ổ bánh và động tác đặt bánh cần kiểm tra dày; khoảng kết thúc còn mở. |
| 38 | synthetic-tkis-038 | L25_V041 | Bảng remember từ khoảng636–688; ví dụ V-ing xuất hiện649; không khẳng định đeo kính, chưa nghe lời giảng về mốc thời gian. |
| 39 | synthetic-tkis-039 | L25_V060 | Sơ đồ đủ tầng1064–1085, đổi slide1088; cần rà biên trong các khoảng lấy mẫu. |
| 40 | synthetic-tkis-040 | L29_V001 | Chất liệu lục bình chưa được nghe xác minh; hai answer dạng hành động khác nhau cần context. |
| 41 | synthetic-tkis-041 | L30_V003 | Kiểm tra playback, đối chiếu nguồn và hai biên. |
| 42 | synthetic-tkis-042 | L24_V035 | Không suy ý định đi giấu hoặc không thấy vật; chỉ giữ hành động nhìn thấy. H3 có context trước answer. |
| 43 | synthetic-tkis-043 | L21_V013 | Đã thấy ba khỉ760–766; nằm trong context riêng sau tê giác. |
| 44 | synthetic-tkis-044 | L24_V044 | Không phân biệt nhảy chủ ý hay té bằng ảnh thưa; cần kiểm tra dày. Thời gian dựa trên video PTS. |
| 45 | synthetic-tkis-045 | L30_V072 | Chi tiết ảnh nhỏ cần crop độ phân giải cao. |
| 46 | synthetic-tkis-046 | L21_V022 | Mô tả nguồn dễ hiểu nhầm hai người và người áo đỏ chung khung hình; thực tế là các shot khác nhau. |
| 47 | synthetic-tkis-047 | L22_V024 | Không suy tốc độ cao từ ảnh; cần clip chuyển động cho hành vi và khoảng hai vòng. |
| 48 | synthetic-tkis-048 | L26_V120 | Thứ tự nguồn không khớp: rau vào chảo trước, phần trắng được cho lại sau; chưa xác minh tên dồi trường chỉ bằng hình. |
| 49 | synthetic-tkis-049 | L26_V392 | Kiểm tra playback, đối chiếu nguồn và hai biên. |
| 50 | synthetic-tkis-050 | L26_V422 | Món trút ra185–187, chuyển chảo trống188; biên cần rà dưới1giây. |
| 51 | synthetic-tkis-051 | L23_V010 | Cần crop để xác nhận chi tiết dây trắng, không đoán đó là tai nghe. |
| 52 | synthetic-tkis-052 | L26_V074 | Toàn bộ nguyên liệu và bảng liệt kê xuất hiện33–39; đổi người dẫn40. Chữ bảng hỗ trợ tên cá thác lác, tôm sú nhưng không gán bằng ngoại hình. |
| 53 | synthetic-tkis-053 | L29_V014 | Chưa xác nhận số lượng mười vật treo hoặc công dụng thớt; H3 là context. |
| 54 | synthetic-tkis-054 | L30_V026 | Kiểm tra playback, đối chiếu nguồn và hai biên. |
| 55 | synthetic-tkis-055 | L23_V017 | Chưa xác nhận countdown 13, tên đường và số người; loại các chi tiết đó khỏi hint. |
| 56 | synthetic-tkis-056 | L25_V060 | Bảng còn1374, đổi slide1377; các nhận định định lượng nguồn chưa đối chiếu từng ô. |
| 57 | synthetic-tkis-057 | L26_V470 | Cắt93–101; nêm125–133; có đoạn băm ngò xen giữa. Chưa xác nhận tiêu trong mẫu nên không giữ chi tiết đó. |
| 58 | synthetic-tkis-058 | L23_V013 | Không suy quốc tịch hay địa điểm từ hình ảnh. |
| 59 | synthetic-tkis-059 | L25_V045 | Sửa áo sọc thành polo đen theo ảnh; slide còn674, chuyển cảnh678. Nền slide sẫm, không phải trắng. |
| 60 | synthetic-tkis-060 | L25_V062 | Hình kéo co thấy551–563, bị ô chữ phủ ở566; không lấy cả thời lượng slide đến577. |
| 61 | synthetic-tkis-061 | L27_V001 | H2 và H3 cung cấp context khác shot, không cùng tồn tại trong answer. |
| 62 | synthetic-tkis-062 | L26_V357 | Không xác định bộ phận thịt hoặc gọi sợi vàng là gừng khi chưa nghe xác minh. |
| 63 | synthetic-tkis-063 | L29_V004 | Công dụng và chất liệu cần thêm lời nói; H3 thuộc context. |
| 64 | synthetic-tkis-064 | L23_V015 | Chưa xác nhận giá trị69 hoặc thứ tự thắng; không đưa vào hint. |
| 65 | synthetic-tkis-065 | L22_V026 | Khoảng nguồn kết thúc400.7 bỏ sót xe tải; H3 nằm ngoài answer nhưng thuộc context; chưa xác nhận cách luồn đòn qua bánh. |
| 66 | synthetic-tkis-066 | L24_V029 | Số hiệu cột trong nguồn chưa đọc được; hình rồng không chứng minh rồng thật đang múa. |
| 67 | synthetic-tkis-067 | L21_V029 | Chưa quan sát được đầy đủ cảnh cầu và xe máy theo query; target chỉ là ứng viên cho moment người phụ nữ, chưa chứng minh đáp ứng toàn query. |
| 68 | synthetic-tkis-068 | L22_V027 | Không xác minh danh hiệu lễ hội lớn nhất; H3 là context. |
| 69 | synthetic-tkis-069 | L23_V008 | Thứ tự vượt chính xác chưa kiểm tra liên tục; đây là khoảng ứng viên rộng cho query nhiều hành động. |
| 70 | synthetic-tkis-070 | L26_V316 | Khoảng nguồn0–324 là gần toàn video; thu hẹp ứng viên. Mẫu10giây chưa đủ xác nhận biên hoặc thứ tự bốn loại rau. |
| 71 | synthetic-tkis-071 | L28_V017 | Không xác định loại phân hoặc sản phẩm trong bao bằng hình ảnh. |
| 72 | synthetic-tkis-072 | L30_V047 | Chưa kiểm chứng chuyển động liên tục để gắn nhãn kiểu bơi; H3 thuộc context. |
| 73 | query-p1-25-kis | L23_V017 | Chưa xác nhận màu áo ở độ phân giải này và giữ vị trí tới đích; ba hint chỉ mô tả cảnh ứng viên, chưa chứng minh toàn query. |
| 74 | tkis-query-02 | L25_V084 | Chỉ mô tả nội dung hiển thị, không kiểm chứng tính đúng của bài giảng; biên cần kiểm tra dày hơn. |
| 75 | tkis-query-12 | L22_V024 | Các chi tiết kiến trúc thấy trong nhiều shot, H3 là context. Không xác minh chất liệu đất nung, địa điểm và cơ chế chống nóng nếu chưa nghe đối chiếu. |
| 76 | source-P1-TRAKE | L26_V200 | Nguồn là TRAKE gồm E1–E4; giữ task_type TRAKE, không tự tính vào benchmark TKIS. Chưa nhìn thấy đĩa trống E1 trong mẫu đầu. Nguồn là TRAKE. Annotation hỗ trợ riêng, chưa chuyển task thành TKIS. |
| 77 | source-P3-query-p3-3-kis | L25_V033 | Video ID nguồn trống; đây là liên kết ứng viên mới từ tìm kiếm và ảnh. Chưa đọc rõ biểu tượng máy bay trên bản đồ. L25_V042 có transcript trùng, cần đối chiếu media đó trước khi kết luận chỉ một video đúng. ASR navigation search; L25_V042 has identical transcript, alternate video pending |

## Reproducibility

- Python 3.13.14, openpyxl 3.1.5, Pillow 12.3.0.
- ffmpeg version 8.1.2 Copyright (c) 2000-2026 the FFmpeg developers
- Code commit khi export: `24a326024511a20d2f4d151913782acd96569c6b`; workspace có thay đổi ngoài task, không commit hoặc sửa các thay đổi đó.
- Hai workbook nguồn không bị sửa; SHA256:
  - `TKIS_queries.xlsx`: `f6d0143cb62fb6aa792a46cc4b2435293a5bd1378626850b54f3d3c4b76d8611`
  - `TKIS_72_queries_PHM_3_hints.xlsx`: `ada04140caec8fec1b1cdcebb38787e03e6d25e14404f0aaefa8bd0af8c3fb85`

Tái xuất sau khi chỉnh visual_notes.py: `backend/.venv/bin/python benchmarks/progressive/annotations/export_annotations.py`, rồi chạy `validate_annotations.py`. Không chạy lại prepare_review.py để ghi đè queue đã mở rộng. Evidence khoảng1.2GB là clip/ảnh cục bộ; không commit toàn bộ media theo mặc định.
