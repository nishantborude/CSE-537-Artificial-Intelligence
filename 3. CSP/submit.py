#do not modify the function names
#You are given L and M as input
#Each of your functions should return the minimum possible L value alongside the marker positions
#Or return -1,[] if no solution exists for the given L

#Your backtracking function implementation

## Class ConsistencyInfo -
## Class to be used to track consistency checks
class ConsistencyInfo:
    numberOfConsistencyChecks = 0
    
    def init(self):
        self.numberOfConsistencyChecks = 0

    def consistencyChecked(self):
        self.numberOfConsistencyChecks += 1

    def getConsistencyCheckCounter(self):
        return self.numberOfConsistencyChecks

## START OF ROUTINES REQUIED FOR BT

# Check if given ruler is Golumb Ruler
def CheckIfGolombRulerExists(addList, consistencyObj):
    lenOfList = len(addList)
    checkSet = []

    #print "Checking for " , addList
    for i in range(0, lenOfList-1):
        for j in range(i+1, lenOfList):
            if addList[j] <= addList[i]:
                return False
            element = addList[j] - addList[i]
            # Consistency counter being incremented
            consistencyObj.consistencyChecked()
            #print i, j, element
            if element in checkSet:
                return False
            else:
                checkSet.append(element)
    return True


# BackTracking Helper Routine 
# Recursive function which assigns each node a MarkerPlace and
# tries for other solutions recursively
def BacktrackHelper(L, M, checkList, curIndex, consistencyObj):
    length = len(checkList)
    newList = list(checkList)
    
    # If placing Last Marker, always place at the end of Ruler
    if curIndex  == M-1:
        newList.append(L)
        if CheckIfGolombRulerExists(newList, consistencyObj):
            return newList
    else:
        for i in range(1, L): # Worst case L but can be tuned to L-M or something like that
            if len(newList) == curIndex:
                newList.append(i)
            else:
                newList[curIndex] = i
            # Check if golomb ruler exists with this Marker placement
            if CheckIfGolombRulerExists(newList, consistencyObj):
                # Try assignment for next Marker
                tList = BacktrackHelper(L, M, newList,curIndex+1, consistencyObj)
                if len(tList) > 0:
                    return tList

    # No Valid Ruler found, Return Empty Ruler 
    return []

## END OF ROUTINES REQUIED FOR BT


## START OF ROUTINES REQUIRED FOR FC

# Initialize Domain
# Initalizes domain in following format
# Domain is list of Markers where each Marker has List of Valid placements and 
# Currently assigned value, by default, un-assigned value is -1
def InitDomain(L, M):
    # DomainList is tuple of (List of Valid marker positions, Choosen position)
    domainList = [[[0], 0]] # Position 0 has these choosen value
    validMarkers = []
    for i in range(1,L):
        validMarkers.append(i)

    for i in range(1, M-1):
        domainList.append([list(validMarkers), -1])
    
    # For last node domain and list is
    domainList.append([[L], L])
    return domainList


# ConvertDomainToMemberList-
# Convert Domain list to memberList which can be quickly verifiable Golomb list
def ConvertDomainToMemberList(domainList):
    memberList = []
    for eachNode in domainList:
        # NOTE: -1 is Marker is not given a position yet
        if eachNode[1] != -1:
            memberList.append(eachNode[1])
    return memberList


# CheckIfGolombRulerExistsFC()
# Forward Checking Approach towards GolombRuler checking
# Removes assignment of current marker from other marker positions
def CheckIfGolombRulerExistsFC(checkList, currentNode, M, consistencyObj):
    memberList = ConvertDomainToMemberList(checkList)
    
    # We can ignore next nodes assignments
    memberList = memberList[:currentNode+1]

    # If such GolombRuler does not exists, Exit
    if not CheckIfGolombRulerExists(memberList,consistencyObj):
        return False
    
    currentNodeInfo = checkList[currentNode]
    currentNodeMarkerPos = currentNodeInfo[1]
    currentNodeDomain = currentNodeInfo[0]

    # Remove Marker position from all other Marker domains
    for i in range(1, len(checkList)-1):
        if currentNodeMarkerPos in checkList[i][0] and checkList[i][1] == -1:
            checkList[i][0].remove(currentNodeMarkerPos)
            # If any Marker's domain becomes empty, then Exit as 
            # this is not recommended position for current marker
            if len(checkList[i][0]) == 0 and i is not currentNode and checkList[i][1] == -1:
                return False
    
    return True

# FindBackTrackingWithSolution()
# Returns Golumb Ruler for given Length L and Marker M
def FindBackTrackingWithFCSolution(L, M, domainList, currentNode, consistencyObj):
    newList = list(domainList)
    if currentNode == M:
        return newList

    #print currentNode, M
    currentNodeInfo = domainList[currentNode]
    for i in currentNodeInfo[0]:
        newList[currentNode][1] = i
        #print 'Before ', newList
        if CheckIfGolombRulerExistsFC(newList, currentNode, M, consistencyObj):
            #print 'Inside ', newList
            # Assignment for current node given, Find recurresively for next node
            tList = FindBackTrackingWithFCSolution(L, M, newList, currentNode+1, consistencyObj)
            if len(tList) > 0:
                return tList

    # No such Golumb Ruler exists, return empty list
    return []


# ForwardCheckingSolution()
# Finds ForwardChecking Solution for given length L and Markers M
def ForwardCheckingSolution(L, M, consistencyObj):
    domainList = InitDomain(L, M)
    
    ansList = FindBackTrackingWithFCSolution(L, M, domainList, 1, consistencyObj)
    return ConvertDomainToMemberList(ansList)


## END OF ROUTINES REQUIRED FOR FC


def BT(L, M):
    "*** YOUR CODE HERE ***"
    consistencyObj = ConsistencyInfo()
    consistencyObj.init()

    # Extreme case
    if L == 0 and M == 1:
        return 1, [0]
    
    # Check if Golumb Ruler exists for given L and M
    golombRuler = BacktrackHelper(L, M, [0], 1, consistencyObj)
    if len(golombRuler) > 0:
        # Find Optimal Golumb Ruler by decreasing L
        prevRuler = list(golombRuler)
        golombRuler1 = list(prevRuler)
        optimalLen = L
        while optimalLen > 0:
            # find optimal golamb ruler now
            prevRuler = list(golombRuler1)
            optimalLen -= 1
            golombRuler1 = BacktrackHelper(optimalLen, M, [0], 1, consistencyObj)
            if len(golombRuler1) == 0:
                break
        print "BT Consistency Check performed: ", consistencyObj.getConsistencyCheckCounter()
        return optimalLen+1, prevRuler
    print "BT Consistency Check performed: ", consistencyObj.getConsistencyCheckCounter()
    return -1,[]

#Your backtracking+Forward checking function implementation
def FC(L, M):
    "*** YOUR CODE HERE ***"
    consistencyObj = ConsistencyInfo()
    consistencyObj.init()
    
    # Extreme case
    if L == 0 and M == 1:
        return 1, [0]

    # Check if Golumb Ruler exists for given L and M
    golumbRuler = ForwardCheckingSolution(L, M, consistencyObj)    
    golumbRuler1 = list(golumbRuler)
    if len(golumbRuler) == M:
        # Find Optimal Golumb Ruler by decreasing L
        prevRuler = list(golumbRuler1)
        optimalLen = L
        while optimalLen >= M:
           prevRuler = list(golumbRuler1)
           optimalLen -= 1
           golumbRuler1 = ForwardCheckingSolution(optimalLen, M, consistencyObj)
           if len(golumbRuler1) != M:
               break
        print "FC Consistency Check performed: ", consistencyObj.getConsistencyCheckCounter()
        return optimalLen+1, prevRuler
    print "FC Consistency Check performed: ", consistencyObj.getConsistencyCheckCounter()
    return -1, []



#Bonus: backtracking + constraint propagation
def CP(L, M):
    "*** YOUR CODE HERE ***"
    return -1,[]

'''
print BT(5, 4)
print FC(5, 4)
print BT(0,1)
print BT(1,1)
print FC(1,1)
print BT(1,2)
print FC(1,2)

print BT(6,4)
print BT(10,4)

print BT(11,5)
print BT(25,7)
print BT(27,7)
print BT(34,8)
#print BT(55,10)
#print BT(72,11)
#print BT(85, 12)

print FC(6,4)
print FC(10,4)


print FC(11,5)
print FC(25,7)
print FC(27,7)

print FC(34,8)
#print FC(55,10)
#print BT(85,12)
'''
