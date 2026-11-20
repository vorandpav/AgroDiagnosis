# Bake VictoriaLogs plugin so defence `up` does not hit the Grafana catalog.
# Install outside /var/lib/grafana so the named data volume cannot hide the plugin.
FROM grafana/grafana:11.6.1

ARG VL_PLUGIN_VERSION=0.21.2
ENV GF_PATHS_PLUGINS=/usr/share/grafana/plugins-bundled

USER root
RUN apk add --no-cache curl \
    && mkdir -p /usr/share/grafana/plugins-bundled \
    && curl -fsSL \
         "https://github.com/VictoriaMetrics/victorialogs-datasource/releases/download/v${VL_PLUGIN_VERSION}/victoriametrics-logs-datasource-v${VL_PLUGIN_VERSION}.zip" \
         -o /tmp/vl-plugin.zip \
    && unzip /tmp/vl-plugin.zip -d /usr/share/grafana/plugins-bundled \
    && rm -f /tmp/vl-plugin.zip \
    && chmod +x /usr/share/grafana/plugins-bundled/victoriametrics-logs-datasource/victoriametrics_logs_backend_plugin_linux_* \
    && chown -R grafana:root /usr/share/grafana/plugins-bundled \
    && apk del curl
USER grafana
