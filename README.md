# Demo de ejecución del Algoritmo Genético para d-MSTr

Este programa ejecuta un ejemplo de búsuqeda de soluciones a una instancia del problema
d-MSTr (árbol de recubrimiento de coste mínimo con restricción de grado y fiabilidad).

## Primer paso

Si estás leyendo esto desde el repositorio en Github, puedes clonar el repositorio ejecutando
```console
git clone https://github.com/frpelaez/dmst.git
cd dmst
```

Si has descargado la carpeta comprimida `dmst.zip`, simplemente descomprímela y entra en ella.

## Requisitos previos a la ejecución

*Nota*: Si tienes `uv` instalado, puedes pasar directamente a la ejecución escribiendo en la consola
```console
uv run demo.py
```

En caso contrario, es necesario tener instalado un intérprete de Python 3.12 o superior. Para comprobarlo, abre la terminal en esta misma carpeta y escribe
```console
python --version
```
Si obtienes un resultado como
```
Python 3.12.x
```
entonces tienes el intérprete de Python instalado.

Una vez tengas instalado el intérprete, debes instalar las dependencias necesarias para la visualización. Simplemente ejecuta
```console
pip install -r requirements.txt
```

## Ejecución de la demo

Para ejecutar la demo y ver los resultados escribe en la terminal
```console
python demo.py
```

## Opciones avanzadas

Es posible ejecutar la demostración sobre un grafo cargado en un fichero CSV. El formato de este archivo ha de ser el siguiente
```csv
// Supón que tienes esto en `datos.csv`
u,v,weight,prob
1,2,50,0.96
1,3,70,0.91
2,3,96,0.92
3,4,110,0.99
3,5,75,0.89
```

Para ello, ejecuta
```console
python demo.py -i datos.csv -o resultados -d_max 4
```

Puedes ver todas las opciones disponibles ejecutando
```console
python demo.py --help
```
