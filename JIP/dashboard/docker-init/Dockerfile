FROM busybox

LABEL IMAGE="tuda-dashboard-init"
LABEL VERSION="0.1.0"

COPY ./db.sqlite3 /db.sqlite3
CMD ["cp", "-u", "/db.sqlite3", "/data/db.sqlite3"]
