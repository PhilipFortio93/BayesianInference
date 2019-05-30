import numpy
import time
from matplotlib import pyplot as plt
from matplotlib import mlab as mlab
from math import ceil as roundup
from mapclass import Map

def fft2D(X):
	X = numpy.fft.fftshift(X)
	return numpy.fft.fft2(X)

def ifft2D(X):
	X = numpy.fft.ifft2(X)
	return numpy.fft.ifftshift(X)

#########
# define rewrap matrix
##############
def wrap(v):
	length = len(v)
	sqrtlength = int(numpy.power(length,0.5))
	twoDt=[[0]*sqrtlength for i in range (sqrtlength)]
	i = 0
	j= 0
	while i < sqrtlength:
		while j < sqrtlength:
			twoDt[i][j]= v[i*sqrtlength+j]
			j = j + 1
		i = i + 1
		j = 0
	twoDt=numpy.array(twoDt)
	return twoDt

############
# define unwrap mamtrix
###############
def unwrap(M):
	M = numpy.array(M)
	rows = M.shape[0]
	columns = M.shape[1]
	t=[0]*rows*columns
	i = 0
	j= 0
	while i < rows:
		while j < columns:
			t[i*rows+j]= M[i][j]
			j = j + 1
		i = i + 1
		j = 0
	return t

###################
# generate a powerspectrum
#####################
def expectedspectrumcreator(sumvector, modklist):
	i=0
	while i < len(modklist)-1: #order such that modklist is monotonically increasing
		if modklist[i] == modklist[i+1]:
			modklist.pop(i)
			sumvector.pop(i)
			i = i - 1
		if modklist[i] > modklist[i+1]:
			a=sumvector[i]
			b=modklist[i]
			sumvector[i]=sumvector[i+1]
			modklist[i]=modklist[i+1]
			sumvector[i+1]=a
			modklist[i+1]=b
			i=-1
		i= i + 1

	i=0
	while i < len(sumvector):
		expectedspectrum.write(str(sumvector[i]))
		expectedspectrum.write(",")
		i = i + 1
	expectedspectrum.write(" ")
	expectedspectrum.write(",")
	i=0
	while i < len(sumvector):
		expectedspectrum.write(str(modklist[i]))
		expectedspectrum.write(",")
		i = i + 1

	expectedspectrum.write("\n")
	#print "newline expected"
	print ("ordered k =", modklist)
	print ("ordered spec =", sumvector)
	return sumvector

###################
# generate a powerspectrum
#####################
def spectrumcreator(sumvector, modklist): #generates a power spectrum from a reduced 2D power spectrum
	i=0
	while i < len(modklist)-1: #order such that modklist is monotonically increasing
		if modklist[i] == modklist[i+1]:
			modklist.pop(i)
			sumvector.pop(i)
			i = i - 1
		if modklist[i] > modklist[i+1]:
			a=sumvector[i]
			b=modklist[i]
			sumvector[i]=sumvector[i+1]
			modklist[i]=modklist[i+1]
			sumvector[i+1]=a
			modklist[i+1]=b
			i=-1
		i= i + 1

	i=0
	while i < len(sumvector):
		spectrum.write(str(sumvector[i]))
		spectrum.write(",")
		i = i + 1
	spectrum.write(" ")
	spectrum.write(",")
	i=0
	while i < len(sumvector):
		spectrum.write(str(modklist[i]))
		spectrum.write(",")
		i = i + 1
	spectrum.write("\n")
	return sumvector
################
# calculate wiener solution
################
def RealSpaceWiener(N,S, datavector):
###########################
# Wiener matrix
##########################
	L=numpy.linalg.inv(S+N)
	W=numpy.dot(S,L)
	W=numpy.matrix(W)
	datavector = numpy.matrix(datavector)
	return numpy.dot(W,datavector.T)

###########################
# define mu of t (messenger field vector)
####################
def MuT(Tval,Nbarval,dval,sval):
	MuT = (Tval*dval)/(Tval+Nbarval)
	MuT = MuT + (Nbarval*sval)/(Tval+Nbarval)
	return MuT
################
def SigmaT(Tval,Nbarval):
	SigmaT = (Tval*Nbarval)/(Tval+Nbarval)
	return SigmaT
#################
def MuS(Sftval,Tftval,tval):
	MuS = (Sftval*tval)/((Sftval+Tftval))
	return MuS
###################
def SigmaS(Sftval,Tftval):
	return (Sftval*Tftval)/(Sftval+Tftval)

if __name__ == '__main__':
	##################
	# OPEN MAP AND READ
	##################
	gridwidth=64
	sigmanoise =0.3
	sigmasignal=4.0
	# mapobj = Map(gridwidth,sigmasignal)
	# mapobj.generateMap()
	# mapobj.addNoise(sigmanoise)
	# mapobj.saveMap()
	print ("Opening...")
	f=open("mapi64", "r") #read in the existing map
	g=open("truemapi64", "r") #true signal map
	print ("Opened and reading")
	
	data1=f.readlines()
	data2=g.readlines()
	f.close()
	g.close()
	print ("Closed")
	map=[[0]*gridwidth for i in range (gridwidth)]
	truemap=[[0]*gridwidth for i in range (gridwidth)]
	n=0
	m=0
	while n<gridwidth: # create map 2D array
		while m < gridwidth:
			map[n][m]=float(((data1[n]).split(","))[m])
			truemap[n][m]=float(((data2[n]).split(","))[m])
			m=m+1
		m=0
		n=n+1
	datavector = unwrap(map) #unwrap noisy map into a data vector
	truesignalvector=unwrap(truemap) #unwrap true signal map into a vector
	print "STOP"
