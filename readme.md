Simplest way to get the code working: download the anaconda python
distribution (https://www.anaconda.com/products/individual). Then you
can run the quantif_synapse.py and then analyse_synapse.py scripts
from Spyder.

The provided scripts are as follows:

* quantif_synapse_cl.py: synapse counting with a command line interface.

* quantif_synapse.py: same as the previous one but with setting script
	variables (use it if you do not know how to use command line).

The input of this script consist in folders with the hierarchy:

analysisDir/batch1/excit/imageStack1.tif
...
analysisDir/batch1/excit/imageStackN.tif
analysisDir/batch1/inhib/imageStack1.tif
...
analysisDir/batch1/inhib/imageStackM.tif
analysisDir/batch2/.....

where "batchNumber" is an integer giving the id of the batch and
imageStack1.tif ... imageStackN.tif are all the stacks for a given
batch and synapse type.

The generate_RGB option allows to generate tiff stacks where synapses
are individually color coded but this step takes extra time and
storage space so is deactivated by default.

Note1: in any case, the script generate tif images containing the
individual synapses, one number for each synapse but this image is not
good for visualisation purposes.

Note2: by default the script does not re-run already generated
analysis, if you want to do it, delete the pickle file or use the
force option.

The output of these scripts are the cluster images (see Note1) as well
as a pickle file (python data storage files) "clusts.pkl", to be
further processesd by other python scripts
(e.g. analyse_synapse_neurons.py)

Note2: the script also detects the different cells using the PV signal
and assign synapses to a neuron if they overlap with any PV signal,
but these information are not used at the moment.


* launch_repeat.sh: a bash script to run multiple times the counting
	script in case of memory issues

* analyse_synapse_neurons.py: output the plots.
