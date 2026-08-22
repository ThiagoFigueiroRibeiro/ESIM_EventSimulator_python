# Convertendo um vídeo em frames

Como preparar um vídeo (`.mp4` e outros formatos que o OpenCV decodifica) para uso com o `esim.cli`.

## 1. Extrair os frames

```bash
python tools/prepare_video.py -i video/video.mp4
```

Isso cria a pasta `video_input/` com:

- `frame_000000.png`, `frame_000001.png`, ... — um PNG por frame do vídeo;
- `images.csv` — índice `timestamp_ns,filename` no formato exigido por `esim.cli` (ver [`FolderImageSource`](../esim/data_provider.py)).

Os timestamps são lidos diretamente do vídeo (`CAP_PROP_POS_MSEC`); se o codec não reportar
timing confiável, o script recorre a um fps fixo automaticamente.

### Opções

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `-i`, `--input` | *(obrigatório)* | Caminho do vídeo (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) |
| `-o`, `--output` | `video_input` | Pasta de destino dos frames e do `images.csv` |
| `--framerate` | — | Sobrescreve o fps do vídeo (usado apenas se o timing por frame não estiver disponível) |

## 2. Gerar os eventos

```bash
python -m esim.cli -i video_input -o demo_out --contrast-threshold 0.2
```

Ajuste `--contrast-threshold` conforme o contraste do vídeo (veja a seção "Executando o
simulador" no [README](../README_Pt-br.md)).

## 3. Visualizar os eventos como frames

```bash
python -m esim.event_frames demo_out/events.txt --output demo_out/event_frames --window-ms 10
```

Gera uma sequência de PNGs (azul = ON, vermelho = OFF) acumulando eventos em janelas de
`--window-ms` milissegundos.