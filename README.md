# Generador de Configuración FluxDev

## Descripción
Aplicación web para generar archivos de configuración YAML para entrenar modelos Fluxdev.

## Requisitos
- Python 3.8+
- Flask
- PyYAML

## Instalación
1. Clonar el repositorio
2. Crear un entorno virtual
3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Ejecución
```bash
python app.py
```

O con Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Configuración en RunPod
1. Clona el repositorio en tu instancia RunPod
2. Configura un firewall para permitir el puerto 5000
3. Ejecuta con Gunicorn

## Licencia
MIT