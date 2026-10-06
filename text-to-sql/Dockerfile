# The recipe's container image, for `render-job` (see README.md, "Running").
#
#   docker build -t <registry>/text-to-sql-recipe:<tag> .
#   docker build --build-arg ACCEPT_VERIEQL_LICENSE=yes -t <registry>/text-to-sql-recipe:<tag> .
#
# The second form also installs VeriEQL under /opt/verieql. VeriEQL is licensed separately, under CC BY-NC-SA 4.0
# (non-commercial use only); building with ACCEPT_VERIEQL_LICENSE=yes accepts that licence. Its runtime manifest's
# path and SHA-256, which `render-job` needs, are written to /opt/verieql.env:
#
#   docker run --rm <image> cat /opt/verieql.env
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# uv builds VeriEQL's own Python 3.11 environment.
RUN pip install "uv==0.12.23"

WORKDIR /opt/text-to-sql-recipe
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# A River client release with the features `train` checks for; --build-arg RIVER_CLIENT=... installs another.
ARG RIVER_CLIENT="river-client>=0.13"
RUN pip install "${RIVER_CLIENT}" ".[verieql,data]"

ARG ACCEPT_VERIEQL_LICENSE=no
RUN if [ "${ACCEPT_VERIEQL_LICENSE}" = "yes" ]; then \
        python -m text_to_sql_recipe setup verieql --accept-license --prefix /opt/verieql > /opt/verieql.env; \
    else \
        echo "# VeriEQL not installed: build with --build-arg ACCEPT_VERIEQL_LICENSE=yes" > /opt/verieql.env; \
    fi

# Data and runs live on the volume `render-job` mounts here.
WORKDIR /workspace
CMD ["python", "-m", "text_to_sql_recipe", "--help"]
