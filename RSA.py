import numpy as np
from numpy.linalg import det, inv
from random import seed
from random import randint
#import sympy.ntheory as nt

# Return list of postions in l where n are.
def postn(n, l):  # This definition uses list comprehension.
    return [x for x in range(len(l)) if l[x] == n]

# Extended Euclidean algorithm
# Bezout Thm
def egcd(a, b):
    s = 0; old_s = 1
    t = 1; old_t = 0
    r = b; old_r = a
    
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    
    #returning gcd, x, y    
    return old_r, old_s, old_t
    
def ModularInv( a, b):
    gcd, x, y = egcd( a, b )
    if x < 0 :
        x += b
    return x

def encrypt( e, n, msg):
    cipher = ""
    
    for c in msg:
        m = ord(c)
        cipher += str(pow(m, e, n)) + " "
        
    return cipher

def decrypt( d, n, cipher): 
    msg = " "
    
    parts = cipher.split()
    for part in parts:
        if part:
            c = int(part)
            msg += chr(pow(c, d, n))
    return msg
    

# From Textbook, Ch 04
# Algorithm 5 Fast Modular Exponentiation Sec 4.2
def modExp(b,n,m) :
    a = "{0:0b}".format(n)
    #print(a)
    x = 1
    power = b%m
    for i in range(len(a)-1,0-1,-1) :
        if a[i] == '1' :
            x = (x*power)%m
    #print(i,":",x)
    power = (power*power)%m
    return x # x = b^n mod m
    

 # MinSet removes repetition of elements in a set.
def MinSet(set) : 
    mnst = []
    for i in range(len(set)) : 
        if not(set[i] in mnst) :
            mnst.append(set[i]) 
    return mnst
    
    
# Prime factorization
def factor(n) : 
    d=2
    factors = [] 
    while n >= d*d :
        if n % d == 0 : 
            n = int(n/d)
            factors.append(d) 
        else :
            d=d+1 
        if n > 1 :
            factors.append(n) 
        return factors
        
# Is n a prime number?
def isPrime(n) :
    return len(factor(n)) == 1
    
# List all divisors of n.
def divisors(n) :
    return [x for x in list(range(1,n+1)) if n%x == 0]
    
# Return quotient and remainder of m/n.
def DivMod(m,n) : 
    return m//n, m%n

# Euler's totient function
def phi(n) :
    fctr_lst = factor(n) 
    distinct_primes = MinSet(factor(n)) 
    ph = 1
    for p in distinct_primes :
        e = len([x for x in fctr_lst if x==p])
        ph *= (p**e-p**(e-1))
    return ph
    
# Return an n-digit format of integer i.
def base10(i,n) :
    return ("{0:0" + str(n) + "d}").format(i)
    
# base10(0,2)
def matrix2list(M) : 
    list = []
    for i in range(np.shape(M)[0]) : 
        for j in range(np.shape(M)[1]) :
            list.append(int(M[i][j])) 
    return list
    
def list2matrix(l,s) : 
    M = np.zeros(s)
    for i in range(s[0]) : 
        for j in range(s[1]) :
            M[i][j] = l[i*s[1]+j] 
    return M
    
def list2text(l) :
    return ''.join([str(alphabetSet[x]) for x in l])

def text2list(s):
    return [postn(s[i], alphabetSet)[0] for i in range(len(s))]
    
def text2numstring(s) :
    return ''.join([base10(postn(s[i], alphabetSet)[0],2) for i in range(len(s))])
    
def num2textstring(s) :
    return ''.join([alphabetSet[10*int(s[2*i])+int(s[2*i+1])] for i in range(len(s)//2)])

# Multiply matrices modulo 73
def mul(X,Y) :
    Z = np.zeros((len(X),len(Y[0]))) # iterate through rows of X
    for i in range(len(X)):
        # iterate through columns of Y
        for j in range(len(Y[0])):
            # iterate through rows of Y 
            for k in range(len(Y)):
                Z[i][j] += int(round(X[i][k] * Y[k][j],0))
    return Z%73
    
# Inverse matrix of M modulo m
def invMatmodm(M,m) :
    s, t, gcd = egcd(m, int(round(det(M),0))) 
    print("M:m =")
    print(M,":",m)
    print("s =",s)
    print("t =",t) 
    print("gcd =",gcd) 
    return t*det(M)*inv(M)%m

def computeGCD(x, y):
  
   while(y):
       x, y = y, x % y
  
   return x

#this is my simple function to find e based on random generation untill gcd( e, phi) == 1
def finde( phi ):
    e = 0
    seed(1)
    while computeGCD(e, phi) != 1:
        e = randint(0, phi / 2)
    return e   
        
# prints 12
print ("The gcd of 60 and 48 is : ",end="")
print (computeGCD(60,48))


print("#")
print("#")
print("#")
print("# Encryption Level 01 starts here:")
print("#")
print("#")
print("#")

alphabetSet = " ABCDEFGHIJKLMNOPQRSTUVWXYZ,.?:'/ 􏰀→-_abcdefghijklmnopqrstuvwxyz0123456789+="
plain_text_level_00 = "MATH 2410 at Webster University Bangkok, Monday 12: 􏰀→30pm//"
#enc = encrypt( e, n, plain_text_level_00)

print("\nalphabetSet is: '"+str(alphabetSet)+"'.") 
print("\nPlain Text is: ", plain_text_level_00 )
plain_text_num = text2numstring(plain_text_level_00)
print("\nPlain Text in number form is: ", plain_text_num)

plain_text_num_list = text2list(plain_text_level_00)
print("\nPlain Text in list form is: ",plain_text_num_list)
#plain_text_num_list = text2list(enc)
#print( plain_text_num_list )

print("\nPlain Text in n-by-3 matrix form is: ", )
M00 = list2matrix(plain_text_num_list, (19,3)) 
print( M00 )

#my chosen martix that is invertable
encodeMatrix = np.array([ [1,2,3], [4,5,6], [7,8,9] ]) # np. 􏰀→identity(3)

print("\na 3-by-3 encoding matrix used is: ") 
cipher_level_01_matrix = mul(M00, encodeMatrix) 
print(cipher_level_01_matrix)

cipher_level_01_list = matrix2list(cipher_level_01_matrix)
print("\ncipher_level_01_list is: ", cipher_level_01_list)

cipher_level_01_text = list2text(cipher_level_01_list)
print("\ncipher_level_01_text is: ", cipher_level_01_text)

cipher_level_01_number = text2numstring(cipher_level_01_text)
print("\ncipher_level_01_number is: ", cipher_level_01_number)
print("#")
print("#")
print("#")
print("# Encryption Level 02 starts here:")
print("#")
print("#")
print("#")
CM01 = ''.join([cipher_level_01_number[i] for i in range(0, 38)])
CM02 = ''.join([cipher_level_01_number[i] for i in range(38, 76)])
CM03 = ''.join([cipher_level_01_number[i] for i in range(76, 114)])
print("\nThe level one encrypted message in number form ")
print("is split into 3 pieces: ")
print("first part:", CM01)
print("second part:", CM02)
print("third part:", CM03)

"""
To see how difficult to decrypt the ciphertext without d, we try another
numerical example using bigger prime numbers.
Example: choose p = 59 649 589 127 497 217 and q = 5 704 689 200 685 129␣
,→054 721
(2 distinct primes).
n = p*q = 340282366920938463463374607431768211457 and
phi(n) = 340282366920938457758625757157511659520.
For simplicity, choose e such that gcd(e,phi(n)) = 1, we choose e =␣
,→10007,
and we found
d = 100483101254259332734759579534370635943.
The company publish {e,n} =␣
,→{10007,340282366920938463463374607431768211457}.
"""
print("\nI have chosen p as a 28-digit prime and q as a 29-digit prime.")

#IMPORTANT - i was testing p, q values and found small values complete the program, but large values cause a string to be out of index 
p = 1066340417491710595814572169
q = 19134702400093278081449423917
#p = 11
#q = 13

#p =  68720001023
#q =  4398050705407

print(p," 'p is prime' is", isPrime(p), ".")
print(q," 'q is prime' is", isPrime(q), ".")
n__ = p * q  # 340282366920938463463374607431768211457
print("\nn is p*q = ", n__, ".")


# sympy.totient(n__) # This phi(n) cannot be easily computed from n:
# phi(n__) # or my version of phi(n).
ph__ = (p - 1) * (q - 1)  # 20404106545895102906154128502005952597176727841607169888
print("\nphi(n) is not feasibly obtained from n,")
print("but it can be easily obtained from p and q as")
print("phi(n)=(p-1)*(q-1) = ", ph__, "(from property of n=p*q).")
print("\nFor simplicity, choose e such that gcd(e,phi(n)) = 1,") 
#using my created function to find an e value 
e__ = finde( ph__) #5981104974851144269057663708755059835153278553042111515
print(" we choose e = ", e__, "and we found")
#finding d using my function ModularInv( a, b):
d__ = ModularInv(e__, ph__) #18003666969718140189380449462570215870930568855332407507
print("d = ", d__, " from the Bezout Thm,")
print("or extended Euclid Algorithm.")

gcd__, s__, t__  = egcd(ph__, e__ )

print("\n(" + str(t__) + ")(" + str(e__) + ")+(" + str(s__) + ")(" + str(ph__) + ") = "
      + str(gcd__) + ".")
      
print("\nIf you mod both sides of the above equation with phi(n),")
print("where phi(n) = 20404106545895102906154128502005952597176727841607169888,")
print("you will see that d*e = 1 (mod phi(n)) or d*e = k*phi(n) + 1,")
print("where k is an integer.")
print("This is the required property for C^d = (M^e)^d = M^(phi(n)*k)*M^1 mod(n),")
print("since the totient function phi(n) has a property that M^phi(n) = 1 (mod n),")
print("then C^d = 1^k*M = M mod(n), and you will recover the message sent over")
print("the internet, M, but this is our level-1-ciphered message.")
print("\nThe company, you, will post only the public key {e, n} and")
print("keep the private key d privately for your own decryption.")
print("\nThe level one ciphertext were split into three chunks:")
M__ = int(CM01)  # ciphertext level 01
print("M01 =", M__, ", and encrypted by the client by the public key,")
C__1 = modExp(M__, e__, n__)  # ciphertext level 02 (encrypted text using public key)
print("and sent over the internet as CM01_L2 =", C__1, ".")
DM__1 = str(modExp(C__1, d__, n__))  # decryption result
C__1 = str(C__1)
# C__1 = '0'*(38-len(C__1))+C__1
DM__1 = str(DM__1)
DM__1 = '0' * (38 - len(DM__1)) + DM__1
print("Once received by you, it then is decrypted by the private key to be DM01 =", DM__1, ".")
M__ = int(CM02)  # ciphertext level 01
print("M02 =", M__, ", and encrypted by the client by the public key,")
C__2 = modExp(M__, e__, n__)  # ciphertext level 02 (encrypted text using public key)
print("and sent over the internet as CM02_L2 =", C__2, ".")
DM__2 = modExp(C__2, d__, n__)  # decryption result
C__2 = str(C__2)
# C__2 = '0'*(38-len(C__2))+C__2
DM__2 = str(DM__2)
DM__2 = '0' * (38 - len(DM__2)) + DM__2
print("Once received by you, it then is decrypted by the private key to be DM02 =", DM__2, ".")
M__ = int(CM03)  # ciphertext level 01
print("M03 =", M__, ", and encrypted by the client by the public key,")
C__3 = modExp(M__, e__, n__)  # ciphertext level 02 (encrypted text using public key)
print("and sent over the internet as CM03_L2 =", C__3, ".")
DM__3 = modExp(C__3, d__, n__)  # decryption result
C__3 = str(C__3)
# C__3 = '0'*(38-len(C__3))+C__3
DM__3 = str(DM__3)
DM__3 = '0' * (38 - len(DM__3)) + DM__3
print("Once received by you, it then is decrypted by the private key to be DM03 =", DM__3, ".")
print("\nThen the three decrypted messages are spliced in postion number format as")
# Cipher_L2 = C__1+C__2+C__3
DM = DM__1 + DM__2 + DM__3
# print(Cipher_L2)
print(DM, ".")

print("In text form it is ", num2textstring(DM), "which is still unreadable, since")

print("it is a level one ciphertext, where its text in list format is")
print(text2list(num2textstring(DM)), ".")
DM_matrix_02 = list2matrix(text2list(num2textstring(DM)), (19, 3))
print("then the list is put in n-by-3 matrix as")
print(DM_matrix_02, ",")
DM_matrix_01 = mul(DM_matrix_02, invMatmodm(encodeMatrix, 73))
print("and decrypted another level by the inverse matrix of the encrypt matrix modulo P=73")
print("to be the matrix of plaintext in postion number format:")
print(DM_matrix_01, ",")
print("next the matrix is converted back to a list of position numbers: ", matrix2list(DM_matrix_01))
print("in which the plaintext sent by the client: ", list2text(matrix2list(DM_matrix_01)))
print("can be read by only you through the list of alphabet characters agreed to be used by")
print("both your shop and your clients: ")
print("'" + str(alphabetSet) + "'.")



