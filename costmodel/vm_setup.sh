#!/bin/bash
# Build patched PG 13.1 + load STATS on a fresh Ubuntu VM (run as the docker-group user).
set -e
D=~/exp/data
mkdir -p "$D/costmodel-scratch"; chmod 777 "$D/costmodel-scratch"
BENCH="$D/End-to-End-CardEst-Benchmark"
[ -d "$BENCH" ] || git clone --depth 1 https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark.git "$BENCH"

echo "=== prep patched PG source ==="
mkdir -p ~/pgbuild && cd ~/pgbuild
[ -f postgresql-13.1.tar.bz2 ] || curl -sSL -o postgresql-13.1.tar.bz2 https://ftp.postgresql.org/pub/source/v13.1/postgresql-13.1.tar.bz2
rm -rf postgresql-13.1; tar xf postgresql-13.1.tar.bz2
( cd postgresql-13.1 && patch -s -p1 < "$BENCH/benchmark.patch" )
tar czf postgres-13.1.tar.gz postgresql-13.1
cp "$BENCH/dockerfile/Dockerfile" "$BENCH/dockerfile/init_pgsql.sh" .
sed -i 's/^FROM debian:buster/FROM debian:bullseye/' Dockerfile

echo "=== docker build (compiles PG; several min) ==="
docker build -t ce-pg131 . >/tmp/build.log 2>&1 && echo "image built" || { echo "BUILD FAILED"; tail -20 /tmp/build.log; exit 1; }

echo "=== run container + load STATS ==="
docker rm -f ce-pg 2>/dev/null || true
docker run -d --name ce-pg -p 5432:5432 -v "$BENCH":/benchmark:ro -v "$D/costmodel-scratch":/scratch ce-pg131
until docker exec ce-pg pg_isready -U postgres >/dev/null 2>&1; do sleep 2; done
docker exec ce-pg psql -U postgres -c "CREATE DATABASE stats;"
docker exec ce-pg psql -U postgres -d stats -f /benchmark/datasets/stats_simplified/stats.sql >/dev/null
for pair in "users users.csv" "posts posts.csv" "postlinks postLinks.csv" "posthistory postHistory.csv" "comments comments.csv" "votes votes.csv" "badges badges.csv" "tags tags.csv"; do
  set -- $pair
  docker exec ce-pg psql -U postgres -d stats -c "COPY $1 FROM '/benchmark/datasets/stats_simplified/$2' WITH (FORMAT csv, HEADER true);"
done
docker exec ce-pg psql -U postgres -d stats -c "ANALYZE;"
echo "=== SETUP DONE; row check ==="
docker exec ce-pg psql -U postgres -d stats -t -c "SELECT 'users '||count(*) FROM users;"
