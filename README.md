# Análisis predictivo de la economía peruana: enero a junio de 2027

Programa en Python que estima las condiciones económicas del Perú para el primer semestre de 2027 con series oficiales del Banco Central de Reserva del Perú (BCRP). Los datos llegan hasta octubre de 2026 (PBI, términos de intercambio y exportaciones, hasta agosto de 2026).

## Cómo ejecutarlo

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
python prediccion_economia_peru.py
```

| Opción | Efecto |
|---|---|
| sin opciones | Corrida completa con los datos de `data/series_bcrp.csv` (unos 25 minutos) |
| `--descargar` | Vuelve a consultar la API del BCRP antes de modelar |
| `--rapido` | Reutiliza los órdenes y la validación guardados en `resultados/` |

## Método

1. **Modelos SARIMA** con búsqueda en malla por AIC y variables de intervención para la pandemia.
2. **Validación de origen móvil** en 24 cortes, frente a un pronóstico ingenuo.
3. **Simulación Monte Carlo** con 5 000 trayectorias para estimar probabilidades.
4. **Determinantes:** regresión con errores de Newey-West y modelos Ridge, bosque aleatorio y *gradient boosting*.
5. **Escenarios** favorable, base y adverso.

## Resultados principales

- PBI, promedio de enero a junio de 2027: alrededor de 3,0 %, con un intervalo al 80 % de 0,6 % a 5,6 %.
- Inflación a 12 meses: cae por efecto de base en marzo de 2027; en junio, el central es de 4,3 %.
- Para llegar a 3 % en junio de 2027, la inflación mensual promedio debe rondar 0,24 %.

## Cómo interpretar los resultados

- El modelo del PBI supera al pronóstico ingenuo. Los de inflación y términos de intercambio no lo superan, por lo que sus centrales son solo referencias.
- Ninguno de los algoritmos de aprendizaje automático superó a la persistencia, así que no se usan como pronóstico.
- Los coeficientes indican asociación, no causalidad.
- Las probabilidades tienen error de simulación y pueden variar uno o dos puntos entre corridas.

## Contenido

- `prediccion_economia_peru.py`: programa principal.
- `data/series_bcrp.csv`: series del BCRP usadas.
- `resultados/`: pronósticos, validación, determinantes, escenarios y resumen.
- `figuras/`: gráficos usados en el documento.
