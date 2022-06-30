rm -rf src/weights/*
echo "Start using pgd to training..."
python src/main_pgd.py
echo "Start using fgm to training..."
python src/main_fgm.py
echo "Complete!" 
