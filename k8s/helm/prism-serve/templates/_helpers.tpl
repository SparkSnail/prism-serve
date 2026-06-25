{{/*
Common helpers for the prism-serve chart.
*/}}

{{- define "prism-serve.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "prism-serve.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "prism-serve.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "prism-serve.labels" -}}
app.kubernetes.io/name: {{ include "prism-serve.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "prism-serve.gateway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "prism-serve.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: gateway
{{- end -}}

{{- define "prism-serve.worker.selectorLabels" -}}
app.kubernetes.io/name: {{ include "prism-serve.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: worker
{{- end -}}

{{- define "prism-serve.gateway.image" -}}
{{- $tag := .Values.gateway.image.tag | default .Chart.AppVersion -}}
{{- printf "%s:%s" .Values.gateway.image.repository $tag -}}
{{- end -}}

{{- define "prism-serve.worker.image" -}}
{{- printf "%s:%s" .Values.worker.image.repository .Values.worker.image.tag -}}
{{- end -}}
