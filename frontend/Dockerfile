FROM docker.m.daocloud.io/library/node:20-alpine AS builder

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./

RUN npm ci

COPY frontend/ .

RUN npm run build


FROM docker.m.daocloud.io/library/nginx:alpine

RUN apk add --no-cache curl

COPY --from=builder /app/dist /usr/share/nginx/html

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]