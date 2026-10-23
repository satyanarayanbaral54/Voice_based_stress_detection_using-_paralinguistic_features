import requests

path = 'archive/Actor_21/03-01-01-01-01-01-21.wav'
with open(path, 'rb') as f:
    r = requests.post('http://127.0.0.1:5000/predict',
                      files={'audio': ('calm_01.wav', f, 'audio/wav')})
print('Status:', r.status_code)
print('Body:', r.text[:2000])
