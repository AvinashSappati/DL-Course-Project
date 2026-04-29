# 1. Base Image
FROM python:3.9

# 2. Set Directory
WORKDIR /app

# 3. Dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Code
COPY . /app/

# 5. THE FIX: Force wget to follow redirects and download the actual 500MB file
# We use -L to follow redirects and --show-progress to see it in the logs
RUN rm -f finalmodel.pth && \
    wget -L -O finalmodel.pth https://github.com/AvinashSappati/DL-Course-Project/releases/download/v1.0.0/finalmodel.pth

# 6. Port Setup
EXPOSE 7860

# 7. Start Server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
