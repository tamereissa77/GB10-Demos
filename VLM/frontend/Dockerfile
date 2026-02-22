# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Vite env vars must be present at build time to be baked into the JS bundle
ARG VITE_APP_MODE=extraction
ENV VITE_APP_MODE=${VITE_APP_MODE}

COPY package.json package-lock.json ./
RUN npm install

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Which nginx config to use: nginx.conf (extraction) or nginx-chat.conf (chat)
ARG NGINX_CONF=nginx.conf

COPY --from=build /app/dist /usr/share/nginx/html
COPY ${NGINX_CONF} /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
