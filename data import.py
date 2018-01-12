
# ### ** : CTR data download **

# #### We use the website click-through data provided by Criteo.  To obtain the data, first accept Criteo's data sharing agreement.  
# #### Below is the agreement from Criteo.  After you accept the agreement, 
# #### you can obtain the download URL by right-clicking on the "Download Sample" button and clicking "Copy link address" or "Copy Link Location", depending on your browser.  
# #### The file is 8.4 MB compressed.  The script below will download the file to the virtual machine (VM) and then extract the data.
# #### If running the cell below does not render a webpage, open the [Criteo agreement](http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/) in a separate browser tab. After you accept the agreement, you can obtain the download URL by right-clicking on the "Download Sample" button and clicking "Copy link address" or "Copy Link Location", depending on your browser.  Paste the URL into the `# TODO` cell below.

# Run this code to view Criteo's agreement
# Note that some ad blocker software will prevent this IFrame from loading.
# If this happens, open the webpage in a separate tab and follow the instructions from above.
from IPython.lib.display import IFrame

IFrame("http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/",
       600, 350)

import glob
import os.path
import tarfile
import urllib
import urlparse

# url should end with: dac_sample.tar.gz
url = "http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz"

url = url.strip()
baseDir = os.path.join('data')
inputPath = os.path.join('cs190', 'dac_sample.txt')
fileName = os.path.join(baseDir, inputPath)
inputDir = os.path.split(fileName)[0]

def extractTar(check = False):
    # Find the zipped archive and extract the dataset
    tars = glob.glob('dac_sample*.tar.gz*')
    if check and len(tars) == 0:
      return False

    if len(tars) > 0:
        try:
            tarFile = tarfile.open(tars[0])
        except tarfile.ReadError:
            if not check:
                print 'Unable to open tar.gz file.  Check your URL.'
            return False

        tarFile.extract('dac_sample.txt', path=inputDir)
        print 'Successfully extracted: dac_sample.txt'
        return True
    else:
        print 'You need to retry the download with the correct url.'
        print ('Alternatively, you can upload the dac_sample.tar.gz file to your Jupyter root ' +
              'directory')
        return False


if os.path.isfile(fileName):
    print 'File is already available. Nothing to do.'
elif extractTar(check = True):
    print 'tar.gz file was already available.'
elif not url.endswith('dac_sample.tar.gz'):
    print 'Check your download url.  Are you downloading the Sample dataset?'
else:
    # Download the file and store it in the same directory as this notebook
    try:
        urllib.urlretrieve(url, os.path.basename(urlparse.urlsplit(url).path))
    except IOError:
        print 'Unable to download and store: {0}'.format(url)

    extractTar()


import os.path
baseDir = os.path.join('data')
inputPath = os.path.join('cs190', 'dac_sample.txt')
fileName = os.path.join(baseDir, inputPath)

if os.path.isfile(fileName):
    rawData = (sc
               .textFile(fileName, 2)
               .map(lambda x: x.replace('\t', ',')))  # work with either ',' or '\t' separated data

print rawData.take(1)
rawDataCount = rawData.count()
print rawDataCount
# This line tests that the correct number of observations have been loadedS
assert rawDataCount == 100000, 'incorrect count for rawData'
if rawDataCount == 100000:
    print 'Criteo data loaded successfully!'

