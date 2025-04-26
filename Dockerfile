# Dockerfile
FROM python:3.12.9-slim

# Thiết lập thư mục làm việc
WORKDIR /backend/app

# Copy và cài dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn FastAPI và user_data (nếu muốn seed sẵn)
COPY app/ ./app
RUN mkdir -p /app/user_data

# Môi trường
ENV DATABASE_URL=postgresql://postgres:nguyenbaduc@db/ai_trading_db
ENV SECRET_KEY="wQ3EX8QjpX0GGE20lyLeSCztyKpSVGYrLafTOihAOCJ3jGUHpzKLhdfcveVa"

# Expose cổng FastAPI
EXPOSE 8000

# Lệnh khởi động
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
