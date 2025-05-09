# Rotten Tomatoes Data Science API

API cho phép lấy thông tin phim, tìm kiếm, gợi ý và dự đoán điểm số của phim dựa trên dữ liệu của Rotten Tomatoes.

## API Endpoints

### 1. Lấy danh sách phim (Pagination)

```
GET /api/movies/?limit={limit}&pagination_token={token}
```

**Tham số:**
- `limit` (1-50): Số lượng phim trả về mỗi trang
- `pagination_token`: Token để lấy trang tiếp theo (null khi không còn trang nào)

**Ví dụ:**
```
GET /api/movies/?limit=10
```

**Luồng xử lý:**
1. Lấy danh sách vector IDs từ Pinecone với limit và pagination token
2. Fetch metadata của từng phim theo ID
3. Trả về danh sách phim và token cho trang tiếp theo

### 2. Lấy thông tin chi tiết phim

```
GET /api/movies/{movie_id}
```

**Tham số:**
- `movie_id`: ID của phim cần lấy thông tin

**Ví dụ:**
```
GET /api/movies/12345
```

### 3. Tìm kiếm phim

```
POST /api/movies/search
```

**Body (JSON):**
```json
{
  "query": "movie title or description",
  "top_k": 10
}
```

**Tham số:**
- `query`: Từ khóa tìm kiếm
- `top_k` (1-50): Số lượng kết quả trả về

### 4. Dự đoán điểm số phim

```
POST /api/movies/predict
```

**Body (JSON):**
```json
{
  "Title": "The Space Adventure",
  "Year": 2023,
  "Duration": "2h15m",
  "Rating": "PG-13",
  "Director": "Christopher Nolan",
  "Synopsis": "A team of astronauts embarks on a dangerous mission to save humanity.",
  "Character1": "John Cooper",
  "Character2": "Dr. Amelia Brand",
  "Character3": "Professor Brand",
  "SciFi": 1,  
  "Adventure": 1,
  "Drama": 1,
  "Thriller": 1,
  "Action": 0,
  "Comedy": 0
}
```

**Kết quả:**
```json
{
  "title": "The Space Adventure",
  "audience_score": 84.5,
  "critics_score": 79.2,
  "explanation": "Phân tích chi tiết về điểm số dự đoán..."
}
```

## Model Training

Mô hình dự đoán được train bằng RandomForest từ dữ liệu phim trên Rotten Tomatoes, sử dụng các yếu tố: 

- Thể loại phim
- Đạo diễn
- Diễn viên/nhân vật
- Thời lượng
- Năm phát hành
- Xếp hạng (G, PG, PG-13, R)

### Quy trình Training:

1. **Thu thập dữ liệu**: Tự động lấy dữ liệu từ API (`model_api_trainer.py`)
   ```
   python -m app.model_api_trainer
   ```

2. **Xử lý dữ liệu**:
   - Chuyển đổi dữ liệu văn bản sang số
   - Xử lý missing values
   - Tạo feature vectors

3. **Huấn luyện mô hình**:
   - Train RandomForest cho điểm audience
   - Train RandomForest cho điểm critics
   - Lưu mô hình (`audience_score_model.pkl`, `critics_score_model.pkl`)
   - Lưu thông tin mô hình (`model_info.pkl`, `weighted_scores.pkl`)

4. **Dự đoán**:
   - Sử dụng ML prediction nếu mô hình hoạt động tốt
   - Fallback sang statistical prediction khi có lỗi

### Files liên quan đến Training:

- `/app/utils/model_trainer.py`: Chứa hàm `train_models()` để huấn luyện mô hình
- `/app/model_api_trainer.py`: Thu thập dữ liệu từ API và gọi train models
- `/app/models/movie_prediction.py`: Định nghĩa class `MoviePredictor` để dự đoán điểm số

### Weights trong Statistical Prediction:

- **Đạo diễn**: 4.0 (hoặc 8.0 nếu điểm > 80% hoặc < 30%)
- **Thể loại**: 2.5 (cố định)
- **Nhân vật/diễn viên**: 2.5 (hoặc 5.0 nếu điểm > 80% hoặc < 30%)
- **Xếp hạng (Rating)**: 1.5
- **Giai đoạn thời gian**: 1.0
