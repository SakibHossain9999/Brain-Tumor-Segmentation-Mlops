#!/usr/bin/env bash
# 5 recorded scenes -> one GIF (both side panels kept), plus a check of length and size
set -e
cd /tmp/gif
W=${1:-960}; FPS=${2:-12}; COLORS=${3:-128}
inputs=""; chain=""
i=0
for f in 1_toon 2_hatching 3_compare 4_slice 5_models; do
  inputs="$inputs -i $f.webm"
  chain="$chain[$i:v]setpts=PTS-STARTPTS,fps=$FPS,scale=1280:720[v$i];"
  i=$((i+1))
done
ffmpeg -v error -y $inputs -filter_complex \
  "${chain}[v0][v1][v2][v3][v4]concat=n=5:v=1,scale=$W:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=$COLORS:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
  demo.gif
echo "gif: $(du -h demo.gif | cut -f1), $(ffprobe -v error -count_frames -show_entries stream=nb_read_frames -of csv=p=0 demo.gif) frames"
