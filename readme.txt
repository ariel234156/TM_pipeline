CRYO-ET PIPELINE - QUICK START
==============================
AreTomo3 / IMOD -> IsoNet2 -> PyTom, driven by one config file.
All explanations: instructions.txt          Questions: Ariel Cohen (coari)

INSTALL
   On moritz: nothing to install. Work in the shared folder:
     cd /mnt/cryonas02/Users/Gabriel_Frank/Ariel_Cohen/aretomo3-isonet2-pipeline

   On another computer:
      git clone https://github.com/ariel234156/TM_pipeline.git
      cd TM_pipeline
      ./install.sh        asks where to install, then installs what's missing

START A NEW DATA SET
   1. ./pipeline new MyData --from 3    (other computers: ./pipeline new MyData)
                                        -> prints your number, e.g. 007
   2. nano configs/007_MyData.yaml      your data, output folders, GPUs
                                        (what to change: instructions.txt, 4)
   3. ./pipeline run 7 --dry-run        must end with "Pipeline finished."
   4. tmux new -s mydata
      ./pipeline run 7
