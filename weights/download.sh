!git clone https://github.com/mohinur2009/traffic_hackathon.git
%cd traffic_hackathon
!bash weights/download.sh
!mkdir -p samples && ln -sf /kaggle/input/*/C3905.MP4 samples/
!python run_submission.py --videos samples --out predictions_samples.json --team test
!python evaluate.py --pred predictions_samples.json --validate-only
