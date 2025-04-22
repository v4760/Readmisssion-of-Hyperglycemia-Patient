# Stage 1: Build React app
FROM node:20-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Stage 2: Serve with Nginx on port 8080
FROM nginx:alpine

# Copy build from previous stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Add custom Nginx config that uses port 8080
RUN echo 'server { \
  listen 8080; \
  server_name localhost; \
  root /usr/share/nginx/html; \
  index index.html; \
  location / { \
    try_files $uri $uri/ /index.html; \
  } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
