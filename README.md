# Resumen del Proyecto IA
Este proyecto es una aplicación que analiza un video para evaluar la comunicación de una persona. El sistema extrae la voz y el lenguaje corporal y genera automáticamente un informe en PDF. El usuario solo tiene que subir un video y la IA hace el análisis

# Estructura de archivos
# 1- app.py
Es la interfaz web con Streamlit.

Qué hace:
Muestra la página al usuario
Permite subir el video
Llama a main.py cuando se pulsa Analizar
Aplica el diseño Retorika (azul y blanco)
Muestra el logo

Es la puerta de entrada del usuario.

# 2-main.py
Es el motor principal del análisis.

Qué hace:
Recibe el video
Extrae el audio
Analiza la voz
Analiza el lenguaje corporal
Genera el archivo informe.pdf

Aquí ocurre toda la inteligencia del proyecto

# 3-retorika_logo.png
Es el logo de la aplicación.

Para qué sirve:
Se muestra en la app web
Se puede usar en el PDF
Da identidad visual Retorika

# 4-.gitignore
Es un archivo de configuración de Git.
