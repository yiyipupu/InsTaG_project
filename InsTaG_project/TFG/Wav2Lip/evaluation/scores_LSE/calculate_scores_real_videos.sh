rm -f all_scores.txt
yourfilenames=`ls $1`

for eachfile in $yourfilenames
do
  python ../syncnet_python/run_pipeline.py --videofile $1/$eachfile --reference wav2lip --data_dir tmp_dir
  python calculate_scores_real_videos.py --videofile $1/$eachfile --reference wav2lip --data_dir tmp_dir >> all_scores.txt
done
