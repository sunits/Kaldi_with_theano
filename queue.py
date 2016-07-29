#!/usr/bin/python

# Queue management system for OAR cluster manager.
# OAR (https://oar.imag.fr/) is an opensource resource manager for
# HPC clusters. It can be used instead of oracle's sungrid (SGE).
# Author: Sunit Sivasankaran
# Instituition : Inria - Nancy

import sys, re, os
import subprocess
import time, random
import pdb

# Default job id file. Should ideally be changed in the prgram
JOB_ID_FILE = ".ARRAY_JOB_IDS"
oarsub_cmd = "oarsub -p \"cluster=\'talc\'\" -q production "
num_threads = "1"
DEBUG = 1



# A hack to have different kaldi installations for GPU and CPU
GPU_path_script = "path_cpu.sh"
path_script = "path.sh"
os.system('cp ' + GPU_path_script + " " + path_script)



class oar_interface:

    MAIN_COMMAND = "oarsub"
    STD_ERR = ""
    STD_OUT = ""
    ARRAY_SIZE = 0
    FINAL_EXEC_COMMAND = ""
    SCRIPT_FILE = ""
    WHITELIST_OPTION = ["-I","-C","-l","--array","--array-param-file","-S","-q","-p","-r","--checkpoint","--signal","-t","-d","--project","-n","-a","--notify","--resubmit","-k","-i","--import-job-key-inline","-e","-O","-E","--hold","-D","-X","-Y","-J"]


    def __init__(self):
# A dictionary to keep tract of all the whitelisted entries
        self.whiteListDict = dict.fromkeys(self.WHITELIST_OPTION)
# A dictionary to keep track of all the options
        self.SUB_OPTION = {}
# Not using more than one core for every job and setting a walltime of 8 hrs
        self.SUB_OPTION["-l"] = "/core=1,walltime=20:00:00"
        self.SUB_OPTION["--array-param-file"] =  JOB_ID_FILE

    def set_stderr(self, err_file):
        self.STD_ERR = err_file
        self.SUB_OPTION["-E"] = err_file

    def set_array_param_file(self, param_file):
        self.SUB_OPTION["--array-param-file"] = param_file

    def set_stdout(self, out_file):
        self.STD_OUT= out_file
        self.SUB_OPTION["-O"] = out_file

    def set_array_count(self, jobStart, jobEnd):
        self.jobStart = jobStart
        self.jobEnd = jobEnd

        os.system("rm -f " + self.SUB_OPTION["--array-param-file"])

        if jobEnd < jobStart:
            print "Jobstart cannot be more than jobEnd"
            exit
        self.SUB_OPTION["--array"] = str(jobEnd - jobStart)
        self.ARRAY_SIZE = jobEnd - jobStart
        with open(self.SUB_OPTION["--array-param-file"],'w') as fid:
            for ele in range(jobStart,jobEnd):
                fid.writelines(str(ele) + "\n")

    def setOption(self, key, value):
        self.SUB_OPTION[key] = value

    def checkOption(self, key):
        try :
            return self.SUB_OPTION[key]
        except :
            return ""

    def set_scriptFile(self, script_file):
        self.SCRIPT_FILE = script_file

# Keep only the whitelisted options, remove the rest
    def cleanOptions(self):
        temp = self.SUB_OPTION.copy()
        for options in temp:
            if not self.whiteListDict.has_key(options):
                self.SUB_OPTION.pop(options)

    def createFinalExecCommand(self):
        self.cleanOptions()
        self.FINAL_EXEC_COMMAND = self.MAIN_COMMAND + " "
        for options in self.SUB_OPTION:
            self.FINAL_EXEC_COMMAND += options + " " + self.SUB_OPTION[options] + " "
        if not self.SCRIPT_FILE:
            print "SCRIPT_FILE not set, Exiting"
            exit
        self.FINAL_EXEC_COMMAND += self.SCRIPT_FILE + " "
        self.FINAL_EXEC_COMMAND += " >> " + self.STD_ERR + " 2>&1 "

    def printExecCommand(self):
        print self.FINAL_EXEC_COMMAND


    def executeScript(self, qDir, arrayLogFile):
        os.system("chmod +x " + self.SCRIPT_FILE)
        ret = os.system(self.FINAL_EXEC_COMMAND)
        logFile = re.sub(r'^--','',self.STD_OUT)

        if not ret == 0:
# Not very sure about this id (256). It fails neverthless
            if ret == 256:
                print sys.argv[0] + " : Error submitting job. Job writing to the log file: " \
                        + logFile + " failed.\n"
            else:
                print sys.argv[0] + " : Error submitting job. \n"
            exit

        # If the job id is not set as environ variable, we will have to get it through means
        #        OAR_JOB_ID = os.environ['OAR_JOB_ID']

        # Obtaining job id from the log file
        OAR_JOB_ID = ""
        with open(self.STD_OUT)  as logFID:
            for logEntry in logFID:
                if  self.ARRAY_SIZE <= 1:
                    if "OAR_JOB_ID" in logEntry:
                        OAR_JOB_ID = logEntry.strip().split("OAR_JOB_ID=")[-1]
                else:
                    if "OAR_ARRAY_ID" in logEntry:
                        OAR_JOB_ID = logEntry.strip().split("OAR_ARRAY_ID=")[-1]

        if OAR_JOB_ID == "":
            print "OAR_JOB_ID could not be found in the log file"
            exit

        oar_job_ctr = 1
        # Check for job competition : TODO
        for jobid  in range(self.jobStart, self.jobEnd):
                wait = 0.1
                doneFile = qDir + "/done." + str(jobid) + "." + OAR_JOB_ID
                while not os.path.isfile(doneFile):
                    time.sleep(wait)
                    wait = wait * 1.2
                    if wait > 3.0:
                        wait = 3.0
                        if random.random() > 0.5 :
                            os.system("touch " + qDir + "/.kick")
                        else:
                            os.system("rm " + qDir + "/.kick 2>/dev/null")
                        os.system("ls " + qDir + " > /dev/null")
                    oar_job_ctr += 1
                    if oar_job_ctr % 10  == 0:
                        if os.path.isfile(doneFile):
                            continue

                        status_file = qDir + "/.status"
                        os.system("rm -f " + status_file)
# For array jobs, there is a job id for every array
                        status_entry = os.system("oarstat -sj " + str(int(OAR_JOB_ID) + jobid - 1) + " > " + status_file);
                        with open(status_file) as status_fid:
                            for status_entry in status_fid :
# Get to the last line of the status file
                                pass

                        os.system("rm -f " + status_file)
# Job does not seem to exists
                        if not re.search(r'[0-9]',status_entry):
                            os.system("touch " + qDir + "/.kick")
                            os.system("rm " + qDir + "/.kick 2>/dev/null");
                            if os.path.isfile(doneFile):
                                continue
                            time.sleep(7)

                            os.system("touch " + qDir + "/.kick")
                            time.sleep(1)
                            os.system("rm " + qDir + "/.kick 2>/dev/null")
                            if os.path.isfile(doneFile):
                                continue

                            time.sleep(60)
                            os.system("touch " + qDir + "/.kick")
                            time.sleep(1);
                            os.system("rm " + qDir + "/.kick 2>/dev/null")
                            if os.path.isfile(doneFile):
                                continue

                            if not re.search(r'\.(\d+)$',doneFile):
                                print "Bad sync-file name"
                                exit(1)

                            array_log_file_check = re.sub(r'${oar_array_param_id}', str(jobid), arrayLogFile)
                            log_last_line = os.popen('tail -n 1 ' + array_log_file_check).read()

                            if re.search(r'status 0$',log_last_line):
                                print "**queue.py: syncfile " + doneFile + " was not created but job seems to have finished OK.  Probably your file-system has problems. This is just a warning.\n";
                                break
                            else:
                                print "**" + sys.argv[0] + ": Error, unfinished job no longer exists, log is in " + array_log_file_check  + ", last line is " + log_last_line + "syncfile is " + doneFile + ", return status of oarstat was " + ret + "\n" + "Possible reasons: a) Exceeded time limit? -> Use more jobs!" + " b) Shutdown/Frozen machine? -> Run again!\n";
                                exit(1);

                        elif re.search(r'Terminated',status_entry):
                            print "queue.py: " + OAR_JOB_ID + " - " + doneFile + " is complete \n"
                            break
                        elif re.search(r'Waiting',status_entry):
                            print "Job in waiting"
                            time.sleep(200)


        allDoneFile = qDir + "/done.*." + OAR_JOB_ID
        os.system('rm -f ' + allDoneFile)

        num_failed = 0
        status = 1
        for jobid  in range(self.jobStart, self.jobEnd):
            array_log_file_check = re.sub(r'\${oar_array_param_id}', str(jobid), arrayLogFile)
            wait_times = [0.1, 0.2, 0.2, 0.3, 0.5, 0.5, 1.0, 2.0, 5.0, 5.0, 5.0, 10.0, 25.0]
            for index, wait in enumerate(wait_times):
                line = os.popen('tail -10 ' + array_log_file_check + ' 2>/dev/null').read()
                if re.search(r'with status (\d+)',line):
                    status = re.search(r'with status (\d+)',line).groups()[0]
                    break
                else:
                    if index < len(wait_times):
                        time.sleep(wait)
                    else:
                        if not os.path.isfile(array_log_file_check):
                            print "Log file " + array_log_file_check + " does not exist"
                        else:
                            print "Last line of the log file : " + array_log_file_check + " does not seem to indicate the return status as expected \n"
                            exit

                if not status == '0':
                   num_failed += 1


        if num_failed == 0:
            exit
        else:
            print sys.argv[0] + ": Jobs failed"



###################################################################################################
# End of class declaration
###################################################################################################




# Create a script file to execute the command
def createScriptFile(cmd, cwd, logFile,oarsub_cmd, syncFile, num_threads, scriptFileName):
    with open(scriptFileName,'w') as scriptFID:
        scriptFID.writelines("#!/bin/bash\n")
        scriptFID.writelines("cd " + cwd + "\n")
        scriptFID.writelines("if [ -z $1 ]; then\n")
        scriptFID.writelines("oar_array_param_id=1\n")
        scriptFID.writelines("else\n")
        scriptFID.writelines("oar_array_param_id=$1\n")
        scriptFID.writelines("fi\n")
        scriptFID.writelines(". ./path.sh\n")
        scriptFID.writelines("( echo '#' Running on `hostname`\n")
        scriptFID.writelines("  echo '#' Started at `date`\n")
        scriptFID.writelines("  echo -n '# '; cat <<EOF\n")
        scriptFID.writelines( cmd + "\n") # this is a way of echoing the command into a comment in the log file,
        scriptFID.writelines("EOF\n") # without having to escape things like "|" and quote characters.
        scriptFID.writelines(") >" + logFile + "\n")
        scriptFID.writelines("time1=`date +\"%s\"` \n")
        scriptFID.writelines(" ( " + cmd + ") 2>>" + logFile + ">>" + logFile + "\n")
        scriptFID.writelines("ret=$?\n")
        scriptFID.writelines("time2=`date +\"%s\"`\n")
        scriptFID.writelines("echo '#' Accounting: time=$(($time2-$time1)) threads=" + num_threads + " >> " + logFile + "\n")
        scriptFID.writelines("echo '#' Finished at `date` with status $ret >> " + logFile + "\n")
        scriptFID.writelines("[ $ret -eq 137 ] && exit 100;\n") # If process was killed (e.g. oom) it will exit with status 137)
          # let the script return with status 100 which will put it to E state) more easily rerunnable.
        scriptFID.writelines(" if [ ! -z $OAR_ARRAY_ID ] ; then ")
        scriptFID.writelines("touch " + syncFile + ".$OAR_ARRAY_ID\n") # touch a bunch of sync-files.
        scriptFID.writelines(" else \n")
        scriptFID.writelines("touch " + syncFile + ".$OAR_JOB_ID\n") # touch a bunch of sync-files.
        scriptFID.writelines(" fi\n")
        #scriptFID.writelines( "touch $syncfile.$OAR_JOB_ID\n") # touch a bunch of sync-files.
        scriptFID.writelines("exit $[$ret ? 1 : 0]\n") # avoid status 100 which grid-engine
#        scriptFID.writelines("## submitted with:\n")       # treats specially.
#        scriptFID.writelines("# oarsub " + cmd + "\n")



###################################################################################################
# Main script begins
###################################################################################################


queue_params = sys.argv
oar = oar_interface()

if len(sys.argv) < 2:
    print "Not enough parameters"
    exit


#TODO: Check if arg_parser can be used to parse the arguments instead

# All jobs are array jobs
param_index = 0
jobname = "JOB"
jobStart = 1
jobEnd = jobStart + 1

# Get config details.
while param_index < len(sys.argv) -2 :
    # index 0 is for file name
    param_index += 1
# Check if the current index points to log file, if so, the rest is the implementaion command
    if ".log" in  queue_params[param_index]:
        logFile = queue_params[param_index]
        param_index += 1
        break

    # For JOB=1:20
    if re.search("^([\w_][\w\d_]*)+=(\d+):(\d+)$",queue_params[param_index]):
            jobname = queue_params[param_index].strip().split("=")[0]
            (jobStart_temp, jobEnd_temp) = queue_params[param_index].strip().split("=")[-1].split(":")
            jobStart = int(jobStart)
            jobEnd = int(jobEnd_temp)
            jobEnd += 1
    # For JOB=1
    if re.search("^([\w_][\w\d_]*)+=(\d+)$", queue_params[param_index]):
            jobname = queue_params[param_index].strip().split("=")[0]
            jobStart = int(queue_params[param_index].strip().split("=")[-1])
            jobEnd = int(jobStart + 1)
    if re.search("^-",queue_params[param_index]):
        key = queue_params[param_index]
        if not re.search(r'^-',queue_params[param_index +1]):
            value = queue_params[param_index +1]
        else:
            value = ""
        oar.setOption(key,value)

if DEBUG:
    print sys.argv



# Creating command file
cmd = ""
for index in range(param_index, len(queue_params) ):
    if not " " in queue_params[index]:
        cmd += queue_params[index] + " "
    elif "\"" in queue_params[index]:
        cmd += "'" + queue_params[index] + "'"
    else:
        cmd += "\"" + queue_params[index] + "\""

    cmd += " "


base_dir = os.path.dirname(logFile)
log_file_name = os.path.basename(logFile)
q_dir =  base_dir + "/q"
q_dir = re.sub(r"(log|LOG)/*q","/q", q_dir)  # If qdir ends in .../log/q, make it just .../q.


queue_logfile = q_dir + "/" + log_file_name

if not os.path.exists(base_dir):
    try:
        os.system("mkdir -p " + base_dir)
    except:
        if not os.path.isdir(base_dir):
            print "Not able to create directory :" + base_dir
            exit

if not os.path.exists(q_dir):
    try:
        os.system("mkdir -p " + q_dir)
        time.sleep(5); ## This is to fix an issue we encountered in denominator lattice creation,
  ## where if e.g. the exp/tri2b_denlats/log/15/q directory had just been
  ## created and the job immediately ran, it would die with an error because nfs
  ## had not yet synced.
    except:
        if not os.path.isdir(q_dir):
            print "Not able to create directory :" + q_dir
            exit

oar.set_array_param_file(q_dir+"/"+JOB_ID_FILE )
oar.set_array_count(jobStart,jobEnd)

logFile = re.sub(jobname,r"${oar_array_param_id}",logFile) # This variable will get
cmd = re.sub(jobname, r'${oar_array_param_id}', cmd) # same for the command...
queue_logfile = re.sub(r'\.?'+jobname,'',queue_logfile) # the log file in the q/ subdirectory
  # is for the queue to put its log, and this doesn't need the task array subscript
  # so we remove it.
queue_scriptfile = queue_logfile

if re.search('\.[a-zA-Z]{1,5}$', queue_scriptfile):
    queue_scriptfile = re.sub(r'\.[a-zA-Z]{1,5}$','.sh', queue_scriptfile)
else:
    queue_scriptfile += ".sh"



cwd = os.getcwd()

if not  re.search(r'^/',queue_scriptfile):
    queue_scriptfile = cwd + "/" + queue_scriptfile

syncFile = q_dir + "/done.${oar_array_param_id}"
os.system("rm " + queue_logfile + " " + syncFile + " " + " 2>/dev/null");


createScriptFile(cmd, cwd, logFile, oarsub_cmd, syncFile, num_threads, queue_scriptfile)
oar.set_stderr(queue_logfile)
oar.set_stdout(queue_logfile)
oar.set_scriptFile(queue_scriptfile)
oar.createFinalExecCommand()
if DEBUG:
    oar.printExecCommand()
oar.executeScript(q_dir,logFile)
