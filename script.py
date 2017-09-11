# -*- encoding: utf-8 -*-
import os
import os.path
import re


urls = []

for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.startswith("http")]:
        urls.append(os.path.join(dirpath, filename))

print len(urls)


regex = r"\<([\s\S]*)>"

with open('corpus.csv', 'w') as output:
    for j, i in enumerate(urls):
        with open(i) as input:
	    if j == 0:
		output.write("url-||-university-||-html-||-target \n")
            search = re.search(regex, input.read())
	    if search: 
		html = search.group(0).replace('\n', ' ').replace('\r', '')
	    else:
		html = 'NaN'
	    target = i.split("/")[2]
	    university = i.split("/")[3]
	    url = i.split("/")[4]
	    print j
	    output.write("{0}-||-{1}-||-{2}-||-{3} \n".format(url, university, html, target))

