# Gerar eventos e gráfico a partir de frames PNG

Este guia explica como usar os frames `.png` e o arquivo `images.csv` da pasta
`video_input` para gerar:

- `events.txt`;
- `events.npz`;
- `preview.png` com a imagem de eventos acumulados e o gráfico da taxa de eventos.

## 1. Conferir a pasta de entrada

A pasta `video_input` deve conter o `images.csv` e todos os frames referenciados
por ele:

```text
video_input/
|-- images.csv
|-- frame_000000.png
|-- frame_000001.png
`-- ...
```

Cada linha do `images.csv` deve possuir um timestamp em nanossegundos e o nome do
arquivo correspondente:

```csv
# timestamp_ns,image
0,frame_000000.png
33333333,frame_000001.png
66666666,frame_000002.png
```

Os comandos abaixo devem ser executados no terminal, a partir da pasta raiz do
projeto.

## 2. Gerar `events.txt` e `events.npz`

No PowerShell, execute:

```powershell
python -m esim.cli `
  --input video_input `
  --output resultado_video `
  --contrast-threshold 0.2
```

O simulador criará uma estrutura semelhante a esta:

```text
resultado_video/
|-- events.txt
|-- events.npz
`-- frames/
```

O arquivo `events.txt` registra um evento por linha no formato `t x y pol`. O
arquivo `events.npz` armazena os mesmos eventos em formato NumPy, nos campos `x`,
`y`, `t` e `pol`.

## 3. Gerar o gráfico `preview.png`

Depois de gerar os eventos, execute:

```powershell
python -m esim.viz resultado_video --save-to resultado_video/preview.png
```

O arquivo será salvo em:

```text
resultado_video/preview.png
```

A visualização contém a imagem dos eventos acumulados e o gráfico da taxa de
eventos ao longo do tempo. Eventos ON aparecem em azul e eventos OFF em vermelho.

## 4. Ajustar a quantidade de eventos

O argumento `--contrast-threshold` controla a sensibilidade da simulação:

- se forem gerados poucos eventos ou nenhum evento, use um valor menor, como
  `0.1`;
- se forem gerados eventos demais, use um valor maior, como `0.3` ou `0.5`.

Exemplo com maior sensibilidade:

```powershell
python -m esim.cli -i video_input -o resultado_video --contrast-threshold 0.1
```

Para evitar misturar arquivos de execuções diferentes, também é possível mudar
o nome da pasta de saída a cada teste:

```powershell
python -m esim.cli -i video_input -o resultado_video_02 --contrast-threshold 0.2
python -m esim.viz resultado_video_02 --save-to resultado_video_02/preview.png
```
