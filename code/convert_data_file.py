"""
@author Sam Ginzburg

This file converts the original nyc_traffic_data.csv into the final data file nyc_traffic_data_final.csv
which we use to perform our final training on.

We perform the following steps
1) Convert street names to Lat/Long coordinate pairs
2) Convert the date column to month and 'day of week' columns (day of week = M-Sun)

Final file format:
|from lat|from long|to lat| to long|direction(NB=0, SB=1, WB=2, EB=3)|month(1-12)|day(M=0, Sun=6)|12:00-1:00 AM|1:00-2:00AM|2:00-3:00AM|3:00-4:00AM|4:00-5:00AM|5:00-6:00AM|6:00-7:00AM|7:00-8:00AM|8:00-9:00AM|9:00-10:00AM|10:00-11:00AM|11:00-12:00PM|12:00-1:00PM|1:00-2:00PM|2:00-3:00PM|3:00-4:00PM|4:00-5:00PM|5:00-6:00PM|6:00-7:00PM|7:00-8:00PM|8:00-9:00PM|9:00-10:00PM|10:00-11:00PM|11:00-12:00AM|


run me:
	python convert_data_file.py --input [input file name].csv --output [output file name].csv
"""


import argparse
import urllib2
import json
import csv
from datetime import datetime


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Converts the input data file into the output data file we are using for this project.')
	parser.add_argument('--input', nargs=1, help='the input file for the program', required=True)
	parser.add_argument('--output', nargs=1, help='the output file for the program', required=True)
	parser.add_argument('--apikey', nargs=1, help='Google Maps API needed for queries', required=True)
	args = parser.parse_args()
	#print args.input[0]
	#print args.output[0]
	passed_count = 0
	with open(args.input[0], "r") as input_file:
		input_file.next() # we want to skip the first line, which is just the header
		output_file = csv.writer(open(args.output[0], 'wb'), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for line in input_file:
			parsed_line = line.split(',')
			output_row = []
			connecting_street = str(parsed_line[2]).strip().replace(" ", "%20") + ',%20NYC'
			street_from = str(parsed_line[3]).strip().replace(" ", "%20") + ',%20NYC'
			street_to = str(parsed_line[4]).strip().replace(" ", "%20") + ',%20NYC'
			try:
				"""
				Lookup streetnames to GPS coordinates, "from"
				"""

				query = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + connecting_street + '%20and%20' + street_from + '&key='+ args.apikey[0]
				response = urllib2.urlopen(query)
				html = response.read()
				json_response = json.loads(html)

				#print json_response['results'][0]['geometry']['location']['lat']
				#print json_response['results'][0]['geometry']['location']['lng']
				output_row.append(json_response['results'][0]['geometry']['location']['lat'])
				output_row.append(json_response['results'][0]['geometry']['location']['lng'])

			except Exception as e:
				print "Unable to retreive GPS coordinates for street"
				print "current line: " + str(parsed_line)
				print "Exception: " + str(e)
				print "json response: " + str(json_response)
				print "query: " + str(query)
				passed_count += 1
				continue

			try:
				"""
				Lookup streetnames to GPS coordinates, "to"
				"""
				query = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + connecting_street + '%20and%20' + street_to + '&key='+ args.apikey[0]
				response = urllib2.urlopen(query)
				html = response.read()
				json_response = json.loads(html)
				#print json_response['results'][0]['geometry']['location']['lat']
				#print json_response['results'][0]['geometry']['location']['lng']
				output_row.append(json_response['results'][0]['geometry']['location']['lat'])
				output_row.append(json_response['results'][0]['geometry']['location']['lng'])

			except Exception as e:
				print "Unable to retreive GPS coordinates for street"
				print "current line: " + str(parsed_line)
				print "Exception: " + str(e)
				print "json response: " + str(json_response)
				print "query: " + str(query)
				passed_count += 1
				continue

			
			
			#print parsed_line[5]
			if 'NB' in parsed_line[5]:
				output_row.append(0)
			elif 'SB' in parsed_line[5]:
				output_row.append(1)
			elif 'WB' in parsed_line[5]:
				output_row.append(2)
			elif 'EB' in parsed_line[5]:
				output_row.append(3)
			else:
				print "Unrecognized direction: " + str(parsed_line[5])
				exit()


			# parse date info
			try:
				datetime_object = datetime.strptime(str(parsed_line[6]), '%m/%d/%y')
				output_row.append(datetime_object.month)
				output_row.append(datetime_object.weekday())
			except Exception as e:
				print "current line: " + str(parsed_line)
				print "Exception: " + str(e)
				exit()

			# add all the times
			output_row.extend(parsed_line[7:])
			
			output_row[len(output_row)-1] = output_row[len(output_row)-1].strip("\r\n")

			output_file.writerow(output_row)

		print "Completed"
		if passed_count > 0:
			print "Skipped " + str(passed_count) + " lines due to errors, see output for details"