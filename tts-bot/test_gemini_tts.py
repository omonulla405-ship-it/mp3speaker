import httpx, sys, io, base64
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
key = 'AQ.Ab8RN6LTzZ3sgsrO7TwFM1FjN5ZuGJDaKtvKehybOWof1eARmg'

# Gemini TTS test
url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash-tts:generateContent?key={key}'
payload = {
    'contents': [{'parts': [{'text': 'Salom! Bu Gemini TTS testi. Ozbek tilida gapiryapman.'}]}],
    'generationConfig': {
        'responseModalities': ['AUDIO'],
        'speechConfig': {
            'voiceConfig': {
                'prebuiltVoiceConfig': {'voiceName': 'Kore'}
            }
        }
    }
}
r = httpx.post(url, json=payload, timeout=60)
print('TTS Status:', r.status_code)
if r.status_code == 200:
    d = r.json()
    parts = d.get('candidates', [{}])[0].get('content', {}).get('parts', [])
    for p in parts:
        if 'inlineData' in p:
            audio = p['inlineData']
            data = audio.get('data', '')
            mime = audio.get('mimeType', '')
            print(f'  Audio: mime={mime}, data_len={len(data)}')
            if data:
                raw = base64.b64decode(data)
                ext = 'wav' if 'wav' in mime else 'mp3'
                fname = f'gemini_tts_test.{ext}'
                with open(fname, 'wb') as f:
                    f.write(raw)
                print(f'  Saved to {fname} ({len(raw)} bytes)')
        elif 'text' in p:
            print(f'  Text: {p["text"][:100]}')
else:
    print('ERROR:', r.text[:400])
