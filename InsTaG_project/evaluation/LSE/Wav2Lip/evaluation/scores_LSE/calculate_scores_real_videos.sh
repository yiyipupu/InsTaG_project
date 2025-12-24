rm -f all_scores.txt

for eachfile in "$1"/*.mp4
do
  python ../syncnet_python/run_pipeline.py \
    --videofile "$eachfile" \
    --reference wav2lip \
    --data_dir tmp_dir

  python calculate_scores_real_videos.py \
    --videofile "$eachfile" \
    --reference wav2lip \
    --data_dir tmp_dir >> all_scores.txt
done
