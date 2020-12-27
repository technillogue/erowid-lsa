import requests
import os
from bs4 import BeautifulSoup

# better: download and extract http://de1.erowid.org/erowid_archives/erowid_mirror_dvd_2012_may_1.iso.bz2

EROWID_BASE_URI = 'https://erowid.org/experiences/exp.php'
EXP_COUNT = 10000

#helper function that takes response body and extracts just the user drug experience
def extract_experience_text(text):
	try:
		begin_delimiter = '<!-- Start Body -->'
		begin = text.index(begin_delimiter) + len(begin_delimiter)
		end = text.index('<!-- End Body -->')
		return text[begin:end].strip()
	except ValueError:
		return ''


for index in xrange(1, EXP_COUNT):

	try:
		data = {'ID': index}

		responseText = requests.get(EROWID_BASE_URI, data).text
		experienceText = extract_experience_text(responseText)
		soup = BeautifulSoup(responseText, "html5lib")
		drug = soup.find('div', {'class': 'substance'}).getText().strip().lower()

		print drug,
		#write experience to folder
		#TODO found problem! people use drug-a/drug-b to denote combination
		#since we plug the string straight in, it thinks that's a nested directory
		#MUST RESCRAPE! 
		folderPath = './experiences/' + drug

		if not os.path.exists(folderPath):
			os.makedirs(folderPath)

		with open('%s/%i.txt' % (folderPath, index), 'w') as outputFile:
			outputFile.write(experienceText)

		print "File written..."

	except:
		print "FAILED"
