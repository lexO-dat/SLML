import requests

def enviar_letra(letra):
    url = "http://localhost:3000/letra"
    payload = {'letra': letra}
    try:
        requests.put(url, json=payload)

    except requests.exceptions.RequestException as e:
        print("Error al enviar la letra:", e)

# Ejemplo de uso:
letra = input("Ingrese la letra que desea enviar: ")  # Solicitar al usuario que ingrese la letra
enviar_letra(letra)
