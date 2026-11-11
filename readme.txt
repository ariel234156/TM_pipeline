CRYO-ET PIPELINE ON MORITZ - QUICK START
========================================
AreTomo3 (or IMOD/RAPTOR) -> IsoNet2 -> PyTom template matching,
driven by one config file. Everything is installed; no conda needed.
Questions: Ariel Cohen (coari).

1. Make your own config (always copy config 3)
   cd /mnt/cryonas02/Users/Gabriel_Frank/Ariel_Cohen/aretomo3-isonet2-pipeline
   ./pipeline new MyDataset --from 3      -> prints your number, e.g. 007

2. Edit it:  nano configs/007_MyDataset.yaml   (or: ./pipeline edit 7)
   - Data:     aretomo.in_prefix, aretomo.gain, aretomo.defect_file
   - Outputs:  change EVERY *dir / output_dir to a folder of yours, and
               pytom.relion5_tomograms_star = <relion5_star.out_dir>/tomograms.star
   - Microscope: aretomo.pix_size, kv, cs, tilt_axis for your session
               (tilt_axis 179.2 is only for the ferritin data). Keep equal:
               pix_size = organize.raw_pixel_size,
               at_bin[0] = organize.bin_factor,
               kv = isonet.prepare_star.voltage
   - GPUs:     check nvidia-smi; use free ones in aretomo.gpu,
               isonet.refine.gpuID, isonet.predict.gpuID, pytom.gpu
   - PyTom:    reference_map, particle_diameter_angstrom,
               voxel_size_angstrom = pix_size x at_bin  (e.g. 1.51 x 8 = 12.08)
               (delete the pytom: section if you don't need it)

3. Dry run (writes nothing, must end with "Pipeline finished."):
   ./pipeline run 7 --dry-run

4. Real run, inside tmux so it survives logout:
   tmux new -s mydata
   ./pipeline run 7          (detach: Ctrl-b d   back: tmux attach -t mydata)

   Part of it only:  --start-from organize   --stop-after aretomo
                     --start-from tm_template  --skip-denoise

5. Results
   aretomo.out_dir             tomograms
   isonet.predict.output_dir   IsoNet2-corrected tomograms
   pytom.out_dir               particles.star, picks_overview.png

Common problems
   "already running"     -> someone runs that config; use your own
   "Permission denied"   -> an output folder is still someone else's
   voxel size mismatch   -> set pytom.voxel_size_angstrom to the value shown
   /mnt/cryonas101       -> not mounted on moritz
   IMOD                  -> keep imod.bin_dir "/home/coari/imod_5.1.13/bin"
                            (/usr/local/IMOD is 5.1.2 and breaks RAPTOR)

Full details: README.md and the comments in your config.
