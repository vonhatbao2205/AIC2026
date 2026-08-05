# HƯỚNG DẪN SỬ DỤNG HỆ THỐNG DRES

⚠️ **Lưu ý:** Nếu tham gia challenge tại SELab (I87), các bạn cần thay đổi hostname của tất cả đường link: `if-wan4.selab.edu.vn` → `10.0.1.21`

## I. GIAO DIỆN VIEWER

- Đăng nhập vào hệ thống tại [http://if-wan4.selab.edu.vn:20740](http://if-wan4.selab.edu.vn:20740) với username và password được cung cấp.
- Các thông tin cần chú ý:
  - Để vào evaluation viewer: nhấn vào biểu tượng hình con mắt ở cột Actions.
  - Evaluation ID (dùng cho việc nộp bài): là giá trị ở cột Run ID.

## II. TÀI LIỆU API & QUY TRÌNH NỘP BÀI

## 1. Địa chỉ giao diện Swagger & Tài liệu API

- **Swagger UI:** [http://if-wan4.selab.edu.vn:20740/swagger-client](http://if-wan4.selab.edu.vn:20740/swagger-client)
- **OpenAPI JSON Spec:** [http://if-wan4.selab.edu.vn:20740/clientapi.json](http://if-wan4.selab.edu.vn:20740/clientapi.json)

## 2. Quy trình nộp bài

### Bước 1: Đăng nhập tài khoản thí sinh (`POST /api/v2/login`)

Thí sinh sử dụng tài khoản được ban tổ chức cấp để đăng nhập và lấy `sessionId`.

- **HTTP Method:** `POST`
- **URL:** `http://if-wan4.selab.edu.vn:20740/api/v2/login`
- **Header:** `Content-Type: application/json`

#### Payload request (JSON):

```json
{
  "username": "<username>",
  "password": "<password>"
}
```

#### Response (HTTP 200):

```json
{
  "id": "<id>",
  "username": "<username>",
  "role": "PARTICIPANT",
  "sessionId": "<sessionId>"
}
```

---

### Bước 2: Lấy chi tiết evaluation run và task đang được mở

#### 2.1. Lấy thông tin evaluation run (`GET /api/v2/client/evaluation/list`)

- **HTTP Method:** `GET`
- **URL:** `http://if-wan4.selab.edu.vn:20740/api/v2/client/evaluation/list?session=<sessionId>`

#### Response (HTTP 200):

```json
[
  {
    "id": "<evaluationId>",
    "name": "<evaluationName>",
    "status": "<evaluationStatus>",
    "taskTemplates": [
      {
        "name": "<taskName>",
        "taskGroup": "<taskGroup>",
        "taskType": "<taskType>"
      }
    ]
  }
]
```

#### 2.2. Lấy chi tiết task đang mở (`GET /api/v2/client/evaluation/currentTask/{evaluationId}`)

- **HTTP Method:** `GET`
- **URL:** `http://if-wan4.selab.edu.vn:20740/api/v2/client/evaluation/currentTask/<evaluationId>?session=<sessionId>`

---

### Bước 3: Nộp đáp án (`POST /api/v2/submit/{evaluationId}`)

- **HTTP Method:** `POST`
- **URL:** `http://if-wan4.selab.edu.vn:20740/api/v2/submit/<evaluationId>?session=<sessionId>`
- **Header:** `Content-Type: application/json`

#### Payload request (JSON):

```json
{
  "answerSets": [
    {
      "taskName": "<taskName>",
      "answers": [
        {
          "mediaItemName": "<videoName>",
          "start": 0,
          "end": 0,
          "text": "<yourAnswer>"
        }
      ]
    }
  ]
}
```

# Download answer for user

GET /download/template/{templateId} trả 200 cho participant và chứa tasks[].targets[] — tức đáp án (media item + temporal range) của mọi task trong template, gồm cả task đang chạy. GET /download/evaluation/{id} cũng chứa template.tasks[].targets. templateId lấy công khai từ client/evaluation/list.

## 3. Ràng buộc các trường trong payload nộp bài

| Trường                    | Kiểu dữ liệu        | Bắt buộc                           | Chi tiết ràng buộc                                                                                           |
| :-------------------------- | :--------------------- | :----------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **`taskName`**      | `String`             | **Bắt buộc**                 | Phải khớp chính xác tên Task đang mở.                                                                    |
| **`mediaItemName`** | `String`             | **Bắt buộc** (với KIS Task) | Tên của file video (Ví dụ:`"L30_V095"` hoặc `"K01_V003"`). **KHÔNG bao gồm đuôi file như `.mp4` |
| **`start`**         | `Integer` / `Long` | **Bắt buộc** (với KIS Task) | Thời gian bắt đầu phân đoạn video (Tính theo mi-li-giây). Phải là số$\ge 0$.                      |
| **`end`**           | `Integer` / `Long` | **Bắt buộc** (với KIS Task) | Thời gian kết thúc phân đoạn video.**Phải lớn hơn hoặc bằng `start`** (`end >= start`).    |
| **`text`**          | `String`             | **Bắt buộc** (với QA Task)  | Câu trả lời cho các task QA.                                                                                |
