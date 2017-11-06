import sys
import random
import numpy as np
import time
def getc1(x, y):
	res= 0
	if x==y:
		res = 0
	elif x<3 and y<3:
		res = 1
	elif x<3 or y<3:
		res = 200
	elif x%7==y%7:
		res = 2
	else:
		res = abs(x-y) + 3
	return res

def getc2(x,y):
	res = 0
	if x==y:
		res = 0
	elif x+y<10:
		res = abs(x-y)+4
	elif (x+y)%11==0:
		res = 3
	else:
		res = abs(x-y)^2+10
	return res

def getc3(x, y):
	if x==y:
		return 0
	else:
		return (x+y)^2

def getMST_cost(graph):
		edges = []
		for i in range(len(graph)-1):
			for j in range(i+1, len(graph)):
				c = getc()
				if c!=0:
					edges.append((c,graph[i],graph[j]))
		sorted(edges)
		mes, mns, mcost = [], {}, 0
		for _ in range(len(edges)):
			c, i, j = edges
			if not(mns.has_key(i) and mns.has_key(j)):
				mes.append((i,j))
				mns[i] = mns[j] = 1
				mcost = mcost + c
		return mes, mcost

class Salesman():
	def __init__(self, n =5, getc = getc1):
		self.n = n
		self.getc = getc
		self.mincost, self.minpath = None, None

	def bruteforce(self):
		def recure(cost, path, pool):
			if len(path) == self.n:
				circost = self.getc(path[0], path[-1])
				if circost != 0:
					if self.mincost is None or  cost + circost< self.mincost:
						self.mincost = cost + circost
						self.minpath = path[:]
			else:
				for i in range(len(pool)):
					rc = self.getc(path[-1], pool[i])
					if rc!=0:
						recure(cost+rc, path+[pool[i]], pool[:i]+pool[i+1:])

		self.mincost = None
		self.minpath = [0]
		recure(0, self.minpath, [i for i in range(1, self.n)])
		if self.mincost:
			print "bruteforce, cost=",self.mincost, ", path=", self.minpath

	def greedy(self, city = None):
		def getnextnearst(city, path):
			moves = []
			for nc in range(0, self.n):
				if nc not in path:
					rc = self.getc(city, nc)
					moves.append((rc, nc))
			moves.sort()
			return moves

		def recure(city, size, path, cost):
			if size == self.n:
		 		self.mincost = cost + self.getc(path[-1], path[0])
		 		self.minpath = path+[path[0]]
			 	return True
			moves = getnextnearst(city, path)
			for rc, nearstcity in moves:
				res = recure(nearstcity, size + 1, path+[nearstcity], cost + rc)
				if res:
					return res
		if city is None:
			city = random.randint(0, self.n-1)
		res = recure(city, 1, [city], 0)
		if res:
			print "greedy, cost=", self.mincost, ":", self.minpath

	def simulatedAnealing(self):
		def getTourLength(tour):
			return reduce(lambda x,y:x+y, [self.getc(tour[i], tour[(i+1)%len(tour)]) for i in range(len(tour))])

		def getFit(tour):
			return 1.0/getTourLength(tour)

		def getRandNeighbor(tour):
			i, j = random.sample(range(len(tour)),2)[:2]
			i, j = min(i,j), max(i, j)
			neighbor = tour[:i]+tour[i:j+1][::-1]+tour[j+1:]
			return neighbor

		def acceptable(currentlen, neighborlen, temprature):
			delta = currentlen - neighborlen
			if delta>0 or random.random()<=np.power(np.e, delta/temprature):
				return True
			return False

		repeat = 10000
		temprature, cooling = 100000, 0.7
		current = range(self.n)
		currentlen = getTourLength(current)
		random.shuffle(current)
		for t in range(1, repeat):
			if temprature==0:
				break
			temprature = temprature * cooling
			randneigbor = getRandNeighbor(current)
			randneigborlen = getTourLength(randneigbor)
			if acceptable(currentlen, randneigborlen, temprature):
				current , currentlen = randneigbor, randneigborlen

		self.mincost, self.minpath = currentlen, current
		print "simulatedAnealing, cost={0}, path:{1}".format(self.mincost, self.minpath)

	def astar(self):
		class Node:
			def __init__(self, path, cost):
				self.path = path
				self.cost = cost

		from heapq import heappush, heappop
		self.mincost, self.minpath = None, []
		current = random.randint(0, self.n)
		heap = [[0, Node([current], 0)]]
		while heap:
			c, node = heappop(heap)
			if len(node.path) == self.n:
				loopcost = node.cost + getc(node.path[-1], node.path[0])
				if self.mincost is None or loopcost<self.mincost:
					self.mincost, self.minpath = loopcost, node.path[:]+[node.path[0]]
			# else:
			# 	for i in range(self.n):
			# 		if i not in node.path:
			# 			g = node.cost + self.getc(node.path[-1], i)
			# 			h = geth(node.path, i)
			# 			heappush(heap, (g+h, Node(node.path[:]+[i], g)))

	def geneticAlg(self):
		class Tour:
			def __init__(self, tour, length = 1):
				self.tour = tour
				self.length = length
				self.fit = 1.0/self.length

			def setProperties(self):
				self.length = getTourLength(self.tour)
				self.fit = 1.0/self.length
			
			def mutate(self):
				index1, index2 = random.randint(0, len(self.tour)-1), random.randint(0, len(self.tour)-1)
				if index1!=index2:
					self.tour[index1], self.tour[index2] = self.tour[index2], self.tour[index1]

		def getTourLength(tour):
			return reduce(lambda x,y:x+y, [self.getc(tour[i], tour[(i+1)%len(tour)]) for i in range(len(tour))])

		def getCrossover(tour1, tour2):
			index = random.randint(0,self.n-1)
			cities = tour1.tour[:index]
			for i in range(index, self.n):
				if tour2.tour[i] not in cities:
					cities.append(tour2.tour[i])
			if len(cities)<self.n:
				for i in range(self.n):
					if tour2.tour[i] not in cities:
						cities.append(tour2.tour[i])
			newtour = Tour(cities)
			return newtour

		def reproduce(tour1, tour2):
			newtour = getCrossover(tour1, tour2)
			if random.random() <= 0.05:
				newtour.mutate()
			newtour.setProperties()
			return newtour

		def getNewPopulation(population):
			def getindex(prob, p):
				s, i = 0, 0
				for i in range(len(prob)):
					if p<=s+prob[i][0]:
						return i
					s = s + prob[i][0]
				return len(prob)-1

			newpopulation = []
			size = len(population)
			normalizefactor = 0
			for tour in population:
				normalizefactor = normalizefactor + tour.fit
			prob = [(tour.fit/normalizefactor, tour) for tour in population]
			for _ in range(size):
				i, j = getindex(prob, random.random()), getindex(prob, random.random())
				newtour = reproduce(population[i], population[j]) if i!=j else Tour(population[i].tour[:], population[i].length)
				newpopulation.append(newtour)
			return newpopulation

		def getInitPopulation(size):
			population = []
			cities = range(self.n)
			for _ in range(size):
				random.shuffle(cities)
				length = getTourLength(cities)
				population.append(Tour(cities[:], length))
			return population

		def rec_regenerate(population, repeat = 250):
			self.mincost, self.minpath = None, None
			for i in range(repeat):
				for tour in population:
					if self.mincost is None or (tour.length<self.mincost):
						self.mincost, self.minpath = tour.length, tour.tour[:]

				population = getNewPopulation(population)

		starttime = time.time()
		populationSize = 4*int(np.log2(n))
		rec_regenerate(getInitPopulation(populationSize))

		print "GA, cost=",self.mincost," path=", self.minpath
		print "exec time=", time.time()-starttime

n= 30
s = Salesman(n, getc1)
# # s.bruteforce()
# s.greedy()
# s.geneticAlg()
s.simulatedAnealing()
# s.astar()
