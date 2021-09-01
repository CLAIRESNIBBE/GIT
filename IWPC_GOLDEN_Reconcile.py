# IWPCModify.py reads in the IWPC ethnicity data set, extracts the Black or African American patients with target INRs
# of 2.5 to enable the generation of a data set which is identical to the IWPC data set used by Innocent Asiimwe, Karen
# Cohen, Munir Pirmohamed et al in their War-PATH paper (2020)
# Claire Snibbe. May 2021
# -----------------------------------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------------------------------
import csv
import pandas as pd
import numpy as np

def main():
    file_name = "IWPCLatest.csv"
    file_name2 = "IWPC_War-PATH_step1.csv"
    file_name3 = "IWPC_War-PATH_step2.csv"
    file_name4 = "IWPC_War-PATH_step3.csv"
    try:
        fileinput = open(file_name, 'r')
        file_content = fileinput.readlines()
        dictIWPC = {}
        if file_content:
            INR_Range = [2.0,2.5,3.0]
            filewritename = input("Enter file name: \n")
            fileoutput = open(filewritename, 'w')
            counter_not_black = 0
            counter_all = 0
            newlines = []
            nextlines = []
            patientsInclude = {}
            with open(file_name, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=";")
                for line in csv_reader:
                    field0 = line[0]
                    field1 = line[4]
                    field2 = line[36]
                    if field1 != "Race (Reported)":
                        counter_all += 1
                        if field1 in dictIWPC:
                            dictIWPC[field1] += 1
                        else:
                            dictIWPC[field1] = 1

                    if field2.find("-") != -1:
                        print(field0, field1, field2)
                        if field2.find(".") == -1:
                            part = field2[0] + ".0" + "-" + field2[2] + ".0"
                            print("date cell to be replaced by", part)

            for race in dictIWPC:
                print(race, dictIWPC[race])
                print(race, dictIWPC[race], file=fileoutput)

            runningtotal = 0
            word1 = "BLACK"
            word2 = "AFRICAN AMERICAN"
            word3 = "AFRICAN-AMERICAN"
            word4 = "OTHER (BLACK BRITISH)"
            for race in dictIWPC:
                runningtotal = runningtotal + dictIWPC[race]
                if (
                        word1 not in race.upper() and word2 not in race.upper() and word3 not in race.upper()) or word4 in race.upper():
                    counter_not_black = counter_not_black + dictIWPC[race]
                    blacklabel = "excluded"
                    if race not in patientsInclude:
                        patientsInclude[race] = False
                else:
                    blacklabel = 'black or african american'
                    if race not in patientsInclude:
                        patientsInclude[race] = True

                print(blacklabel, race, dictIWPC[race], file=fileoutput)

            for race in patientsInclude:
                print(race, patientsInclude[race])

            print('Total is ', runningtotal, file=fileoutput)
            print('Excluded patients are ', counter_not_black, file=fileoutput)
            print('Number of patients in original IWPC ethnicity data set is ', counter_all)
            with open(file_name, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=";")
                for line in csv_reader:
                    field0 = line[0].strip()
                    field1 = line[4].strip()
                    field2 = line[36].strip()
                    if field1 != "Race (Reported)":
                        indexPos = patientsInclude.get(field1)
                        if indexPos == True:
                            newlines.append(line)
                    else:
                        newlines.append(line)

            with open(file_name2, 'w', newline='') as new_file:
                csv_writer = csv.writer(new_file, delimiter=";")
                counter = 0
                NAcounter = 0
                lines_revised = []
                lines_modified = []
                for line in newlines:
                    field0 = line[0].strip()
                    field1 = line[4].strip()
                    field2 = line[36].strip()
                    field3 = line[35].strip()
                    if field1 != "Race (Reported)":
                        counter = counter + 1
                    print(field0, field1, field2, field3)
                    if field2.find("-") != -1:
                        print(field0, field1, field2, field3)
                        if field2.find(".") == -1:
                            part = field2[0] + ".0" + "-" + field2[2] + ".0"
                            print("date cell to be replaced by", part)
                            field2 = part.strip()
                            line[36] = field2
                            print(field0, field1, field2, field3)
                    if field3 == "NA":
                        NAcounter = NAcounter + 1
                        if field2.find("-") != -1 and field2 != "NA":
                            print("Target INR is NA or Unknown and Estimated Target INR is ", field2)
                            line[35] = field2
                            lines_modified.append(line)
                    lines_revised.append(line)
                    csv_writer.writerow(line)

                rev_counter = 0
                rev_NAcounter = 0

                for line in lines_revised:
                    field0 = line[0].strip()
                    field1 = line[4].strip()
                    field2 = line[36].strip()
                    field3 = line[35].strip()
                    if field1 != "Race Reported":
                        rev_counter += 1
                    if field3 == "NA":
                        rev_NAcounter = rev_NAcounter + 1
                        print(rev_NAcounter, "Still NA ", field0, field1, field3, field2)

                print('No of patients in original IWPC File is ' + str(counter_all))
                print('No of patients not of reported race=Black or African American ', str(counter_not_black))
                print('No of patients in IWPC File reflecting purely reported race=Black or African American is ', counter)
                print('No of NA records in IWPC File reflecting purely reported race=Black or African American is ' + str(NAcounter))
                print('No of patients in Target INR column changed from NA ', len(lines_modified))
                print('No of NA patients in revised IWPC File is ', rev_NAcounter)




                count_recorder = 0
                for line in lines_modified:
                    count_recorder = count_recorder + 1
                    field0 = line[0].strip()
                    field1 = line[4].strip()
                    field2 = line[36].strip()
                    field3 = line[35].strip()
                    print(count_recorder, field0, field1, field2, field3)

        fileinput2 = open(file_name2, 'r')
        file_content2 = fileinput2.readlines()
        INR_Invalid=[]
        INR_NA = []
        WarDose_NA = []
        count_WarDoseNA = 0
        if file_content2:
            with open(file_name2, 'r') as csv_file:
                csv_reader2 = csv.reader(csv_file, delimiter=";")
                countTargetINR = 0
                countINR = 0
                countTargetINR_NA = 0
                countINR_NA = 0

                for line in csv_reader2:
                    field0 = line[0].strip()
                    field1 = line[35].strip()  # Target INR
                    field2 = line[36].strip()  # Estimated Target INR Range
                    field4 = line[38].strip()  #Therapeutic Dose of Warfarin

                    if not (field1 == "Target INR"):
                       if field1 != "NA":
                         if field1.find('-') == -1:
                           floatstring1 = float(field1.replace(',', '.'))
                           if not floatstring1 in INR_Range:
                             INR_Invalid.append(line)
                             countTargetINR += 1
                           else:
                             dashpos = field2.find("-")
                             fieldlen = len(field2)
                             lowerlimit = field1[0:dashpos]
                             upperlimit = field1[dashpos + 1:fieldlen]
                             lowerlimit = lowerlimit.strip()
                             upperlimit = upperlimit.strip()
                             lowerlimit = lowerlimit.replace(',','.')
                             upperlimit = upperlimit.replace(',','.')
                             if lowerlimit != '' and upperlimit != '':
                               if not (float(lowerlimit) in INR_Range) and not (float(upperlimit) in INR_Range):
                                 INR_Invalid.append(line)
                                 countTargetINR += 1
                       else:
                            if field1 == "NA":
                              if field2 != "Estimated Target INR Range Based on Indication":
                                if field2 != "NA":
                                  if field2.find("-") != -1:
                                    dashpos = field2.find("-")
                                    field2 = field2[0:dashpos]
                                    floatstring2 = float(field2.replace(',', '.'))

                                  if not (floatstring2 in INR_Range):
                                    countTargetINR += 1  # then the INR for this patient is truly NA!
                                else:
                                  countTargetINR_NA += 1
                                  INR_NA.append(line)



            print("Number of invalid INRs in TARGET INR Column ", countTargetINR)
            print("Number of NAs in TARGET INR Column ", countTargetINR_NA)
            counter = 0
            INR_NA_ID = []
            INR_Invalid_ID = []
            WAR_NA_ID = []
            for line in INR_Invalid:
                counter = counter+1
                field0 = line[0].strip()
                if field0 != 'PharmGKB Subject ID' and not field0 in INR_Invalid_ID:
                  INR_Invalid_ID.append(field0)
                  field1 = line[4].strip()
                  field2 = line[36].strip()
                  field3 = line[35].strip()
                  print(counter, field0, field1, field2, field3)

            counter = 0
            for line in INR_NA:
                counter = counter + 1
                field0 = line[0].strip()
                if field0 != 'PharmGKB Subject ID' and not field0 in INR_NA_ID:
                  INR_NA_ID.append(field0)
                  field1 = line[4].strip()
                  field2 = line[35].strip()
                  print(counter, field0, field1, field2)



            counter = 0
            for line in WarDose_NA:
                counter = counter + 1
                field0 = line[0].strip()
                if field0 != 'PharmGKB Subject ID' and not field0 in WAR_NA_ID:
                    WAR_NA_ID.append(field0)
                    field1 = line[4].strip()
                    field2 = line[38].strip()
                    print(counter,field0,field1,field2)


            nextlines = []
            declines = []
            with open(file_name3, 'w', newline='') as new_file:
                csv_writer = csv.writer(new_file, delimiter=";")
                print("Number of patients in newlines ", len(newlines))
                for line in newlines:
                    field0 = line[0].strip()
                    if field0 != 'PharmGKB Subject ID' and (field0 in INR_NA_ID or field0 in INR_Invalid_ID ):
                      print(field0, "excluded")
                      declines.append(line)
                    else:
                       nextlines.append(line)
                       csv_writer.writerow(line)
            print('No. of patients with an invalid INR ',  len(INR_Invalid_ID))
            print('Therefore, the no. of patients with an invalid INR or an INR of NA is ', len(declines))
            print('and the no of patients with a Target INR of 2.5 or 3.0 is ', len(nextlines)-1) #excluding row header
            counter = 0
            for line in declines:
                counter = counter+1
                field0 = line[0].strip()
                field1 = line[4].strip()
                field2 = line[35].strip()
                print(counter, field0, field1, field2)

        fileinput3 = open(file_name3, 'r')
        file_content3 = fileinput3.readlines()
        lastlines = []
        WarDose_NA = []
        count_WarDoseNA = 0
        if file_content3:
            with open(file_name3, 'r') as csv_file:
                csv_reader3 = csv.reader(csv_file, delimiter=";")
                for line in csv_reader3:
                   field0 = line[0].strip()
                   field1 = line[4].strip()
                   field2 = line[38].strip()
                   if field2 != "NA":
                      lastlines.append(line)
                   else:
                      if field2 != "Therapeutic Dose of Warfarin":
                         count_WarDoseNA += 1
                         WarDose_NA.append(line)



            print("Number of patients with NA as their Therapeutic Dose of Warfarin ", counter)
            print("Final number of patients in IWPC File ",len(lastlines)-1) #excluding row header

            with open(file_name4, 'w', newline='') as new_file:
                csv_writer = csv.writer(new_file, delimiter=";")
                for line in lastlines:
                   csv_writer.writerow(line)



    except FileNotFoundError:
        print("Sorry, could not find file " + file_name)


if __name__ == "__main__":
    main()