# Use official Python image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy all files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install gunicorn

# Expose the port your app will run on
EXPOSE 5000

# Start the app using Gunicorn (production server)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]
