#!/bin/sh
# The official PostgreSQL image sources non-executable .sh init files on first init.
# This role is created only when the data directory is empty; changing .env later
# does not rotate an already-created database role.
set -eu

if [ "$APP_DB_USER" = "$POSTGRES_USER" ]; then
    echo >&2 "APP_DB_USER must differ from the PostgreSQL bootstrap/admin POSTGRES_USER"
    return 1
fi

psql \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --set=ON_ERROR_STOP=1 \
    --set=admin_user="$POSTGRES_USER" \
    --set=app_user="$APP_DB_USER" \
    --set=app_password="$APP_DB_PASSWORD" \
    --set=app_db="$POSTGRES_DB" <<'SQL'
CREATE ROLE :"app_user" LOGIN PASSWORD :'app_password';
GRANT CONNECT ON DATABASE :"app_db" TO :"app_user";
GRANT USAGE ON SCHEMA public TO :"app_user";
ALTER DEFAULT PRIVILEGES FOR ROLE :"admin_user" IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"app_user";
ALTER DEFAULT PRIVILEGES FOR ROLE :"admin_user" IN SCHEMA public
    GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO :"app_user";
SQL
