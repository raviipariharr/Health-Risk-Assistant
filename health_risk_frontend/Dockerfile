# ============================================================
# Dockerfile — AI Health Risk Assistant (Frontend)
# ============================================================
#
# Our frontend is plain HTML/CSS/JS — no build step needed.
# We just need a web server to serve the static files.
#
# nginx is the industry-standard static file server.
# The official nginx:alpine image is only ~23MB.
#
# Build:  docker build -t health-risk-frontend .
# Run:    docker run -p 3000:80 health-risk-frontend
# ============================================================

FROM nginx:alpine

# Copy our static files into nginx's serving directory
COPY . /usr/share/nginx/html

# Custom nginx config — handles SPA routing and sets CORS headers
# (so the frontend can call the API from a different port/domain)
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
\n\
    # Serve files directly, fall back to index.html\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
\n\
    # Security headers\n\
    add_header X-Frame-Options "SAMEORIGIN";\n\
    add_header X-Content-Type-Options "nosniff";\n\
    add_header Referrer-Policy "strict-origin-when-cross-origin";\n\
\n\
    # Cache static assets aggressively, HTML not at all\n\
    location ~* \.(css|js|png|jpg|ico|woff2)$ {\n\
        expires 1y;\n\
        add_header Cache-Control "public, immutable";\n\
    }\n\
    location ~* \.html$ {\n\
        expires -1;\n\
        add_header Cache-Control "no-cache";\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget -qO- http://localhost/ || exit 1