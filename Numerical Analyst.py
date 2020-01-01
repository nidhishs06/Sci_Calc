# Importing necessary libraries and modules
from math import *
import numpy as np
import sympy as sy
from decimal import Decimal
from cmath import sqrt as cqrt
import matplotlib.pyplot as pl
import os


# CONTROL FLOW STATEMENTS
def CONTINUE():
    ch = 'X'
    while ch not in 'YN':
        print("Do you want to continue? [Y/N]")
        ch = input()
        ch = ch.upper()

    if ch == 'Y':
        return 1
    if ch == 'N':
        return 0


def EXIT():
    print("Ending Numerical Analyst..")


# End -- CONTROL FLOW STATEMENTS


# 1. Basic Arithmetic Operations--------------------
def BAO():
    print("\n\n\n")
    print("Choose Operation: ")
    print("a. Quick Calculate")
    print("b. Consecutive Addition")
    print("c. Consecutive Multiplication")
    print("d. Division")
    print("e. Powers")
    print("f. Logarithm")
    print("g. Factorials")
    print("h. Combinations")
    print("i. Permutations")
    ch = '0'
    # Accept valid input within bounds
    while ch not in 'abcdefgh':
        ch = input("Enter valid choice [a-h]: ")
        ch = ch.lower()

    # Execute proper operation
    if ch == 'a': qCALC()
    if ch == 'b': cADD()
    if ch == 'c': cMUL()
    if ch == 'd': DIV()
    if ch == 'e': EXP()
    if ch == 'f': LOG()
    if ch == 'g': FACT()
    if ch == 'h': COMB()
    if ch == 'i': PERM()


def qCALC():
    print("\n\nAnswer will be calculated according to the BEDMAS Rule:")
    cc = 1
    # Handling exceptions due to eval()
    while cc:
        try:
            ans = eval(input("Enter the sequence of numbers with appropriate operators.\n"))
            cc = 0
        except:
            print("Wrong Input! Please enter a valid numerical input.\n")
            cc = 1
    print("Answer: ", ans)
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("quickCALC: " + str(ans) + "\n")

    if CONTINUE():
        qCALC()
    else:
        MMenu()


def cADD():
    print("\n\n")
    try:
        n = int(input("How many numbers do you want to add? "))
    except:
        print("Enter valid numerical input")
        cADD()
    print("Enter the list of numbers to calculate the sum.")
    sum = 0
    try:
        for i in range(n):
            sum += int(input())
    except:
        print("Enter valid numerical input.")
        cADD()

    print("SUM:", sum)
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("consecutiveADD: " + str(sum) + "\n")

    if CONTINUE():
        cADD()
    else:
        MMenu()


def cMUL():
    print("\n\n")
    try:
        n = int(input("How many numbers do you want to multiply? "))
    except:
        print("Enter valid numerical input")
        cMUL()

    print("Enter the list of numbers to calculate the product.")
    pr = 1
    try:
        for i in range(n):
            pr *= float(input())
    except:
        print("Enter valid numerical input")
        cMUL()
    print("PRODUCT:", pr)
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("consecutiveMULTIPLY: " + str(pr) + "\n")

    if CONTINUE():
        cMUL()
    else:
        MMenu()


def DIV():
    print("\n\n")
    try:
        d1 = float(input("Enter dividend: "))
        d2 = float(input("Enter divisor: "))
    except:
        print(("Enter valid numerical input"))
        cMUL()
    try:
        ans = divmod(d1, d2)
    except:
        print("Division by zero error.")
        if CONTINUE():
            DIV()
        else:
            MMenu()


    print("Quotient:", ans[0], "\nRemainder:", ans[1], "\nDivision:", (d1 / d2))
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("DIV: " + str(d1 / d2) + "\n")

    if CONTINUE():
        DIV()
    else:
        MMenu()


def EXP():
    print("\n\n")
    # Handling float exceptions
    try:
        b = float(input("Enter base:"))
        pow = float(input("Enter power:"))
    except:
        print("Enter numerical values only")
        EXP()
    ans = b ** pow
    print("Result:", ans)
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("EXP: " + str(ans) + "\n")

    if CONTINUE():
        EXP()
    else:
        MMenu()


def LOG():
    print("\n\n")
    # Handling Logarithm definition conditions
    try:
        while True:
            b = input("Enter valid base: [e/10]")
            if b in ['e', '10']:
                break
        while True:
            N = float(input("Enter valid number:"))
            if N > 0:
                if b == e:
                    ans = np.log(N)
                else:
                    ans = np.log10(N)
                    ans = round(ans, 4)

    except:
        print("Enter numerical values only.")
        LOG()

    print("Logarithm: ", ans)
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("LOG: " + str(ans) + "\n")

    if CONTINUE():
        LOG()
    else:
        MMenu()


def FACT():
    print("\n\n")
    # Handling factorial definition conditions
    try:
        while True:
            N = int(input("Enter positive integral to calculate factorial: "))
            if N >= 0:
                break
    except:
        print("Enter numerical values only!")
        FACT()

    fact = 1
    for i in range(1, N + 1):
        fact *= i
    print("The factorial:", fact)
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("FACTORIAL: " + str(fact) + "\n")

    if CONTINUE():
        FACT()
    else:
        MMenu()


def COMB():
    print("\n\n")
    try:
        while True:
            n = int(input("n? "))
            if n > 0:
                break
        while True:
            r = int(input("r? "))
            if 0 < r < n:
                break
    except:
        print("Enter positive integral values only.")
        COMB()

    nCr = factorial(n) / (factorial(r) * factorial(n - r))
    print("nCr: ", nCr)
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("nCr: " + str(nCr) + "\n")

    if CONTINUE():
        COMB()
    else:
        MMenu()


def PERM():
    print("\n\n")
    try:
        while True:
            n = int(input("n? "))
            if n > 0:
                break
        while True:
            r = int(input("r? "))
            if 0 < r < n:
                break
    except:
        print("Enter positive integral values only.")
        PERM()

    nPr = factorial(n) / (factorial(n - r))
    print("nPr: ", nPr)
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("nPr: " + str(nPr) + "\n")

    if CONTINUE():
        COMB()
    else:
        MMenu()


# END -- 1. Basic Arithmetic Operations---------------


# 2. Trigonometric Function --------------------------
def TF():
    print("\n\n\n")
    print("Choose Operation: ")
    print("a. sin")
    print("b. cos")
    print("c. tan")
    print("d. arcSin")
    print("e. arcCos")
    print("f. arcTan")

    while True:
        ch = input("Enter valid choice [a-f]: ")
        ch = ch.lower()
        if ch in 'abcdef':
            break

    m = 'R'
    while True and ch in 'abc':
        m = input("Select input mode. Degrees or Radians? [D/R]")
        m = m.upper()
        if m in 'DR':
            break

    try:
        if ch in 'abc':
            x = eval(input("Enter angle: "))
        else:
            x = eval(input("Enter input: "))
        if m == 'D' and ch in 'abc':
            x = radians(x)

        # Execute proper operation
        if ch == 'a':
            ans = np.sin(x)
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("sin: " + str(ans) + "\n")
        if ch == 'b':
            ans = np.cos(x)
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("cos: " + str(ans) + "\n")
        if ch == 'c':
            ans = np.tan(x)
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("tan: " + str(ans) + "\n")
        if ch == 'd':
            ans = degrees(np.arcsin(x))
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("arcSin: " + str(ans) + "\n")
        if ch == 'e':
            ans = degrees(np.arccos(x))
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("arcCos: " + str(ans) + "\n")
        if ch == 'f':
            ans = degrees(np.arctan(x))
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("arcTan: " + str(ans) + "\n")
    except:
        print("Enter valid numerical input only")
        TF()

    print(ans)

    if CONTINUE():
        TF()
    else:
        MMenu()


# END -- 2. Trigonometric Function -------------------


# 3. Matrices and Determinants -----------------------
def MAT(A=sy.Matrix(3, 3, [0, 0, 0, 0, 0, 0, 0, 0, 0]), B=sy.Matrix(3, 3, [0, 0, 0, 0, 0, 0, 0, 0, 0])):
    while True:
        print("\n\n\n")
        print("Choose Operation: ")
        print("a. Enter MatA")
        print("b. Enter MatB")
        print("c. Edit Mat")
        print("d. Addition")
        print("e. Subtraction")
        print("f. Multiplication")
        print("g. Determinant")
        print("h. Inverse")
        print("i. Adjoint")

        while True:
            ch = input("Enter valid choice [a-i]: ")
            ch = ch.lower()
            if ch in 'abcdefghi':
                break

        if ch == 'a':
            A = createMAT()
        if ch == 'b':
            B = createMAT()
        if ch == 'c':
            while True:
                ch1 = input("Which matrix do you want to edit? [A/B]: ")
                ch1 = ch1.lower()
                if ch1 in 'ab':
                    break
            if ch1 == 'a':
                A = createMAT()
            if ch1 == 'b':
                B = createMAT()
        if ch == 'd':
            ans = A + B
            print(ans)
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("MatA+MatB: " + str(ans) + "\n")
            break
        if ch == 'e':
            print("\n")
            print("1. A-B")
            print("2. B-A")
            while True:
                ch1 = input("Enter valid choice [1/2]: ")
                if ch1 in '12':
                    break
            if ch1 == '1':
                ans = A - B
                with open("AnsMemory.txt", 'a') as fh:
                    fh.write("MatA-MatB: " + str(ans) + "\n")
            if ch1 == '2':
                ans = B - A
                with open("AnsMemory.txt", 'a') as fh:
                    fh.write("MatB-MatA: " + str(ans) + "\n")
            print(ans)
            break
        if ch == 'f':
            print("\n")
            print("1. MatAxMatB")
            print("2. MatBxMatA")
            print("3. MatAxMatA")
            print("4. MatBxMatB")
            while True:
                ch1 = input("Enter valid choice [1-4]: ")
                if ch1 in '1234':
                    break
            if ch1 == '1':
                ans = A * B
                with open("AnsMemory.txt", 'a') as fh:
                    fh.write("MatA*MatB: " + str(ans) + "\n")
            if ch1 == '2':
                ans = B * A
                with open("AnsMemory.txt", 'a') as fh:
                    fh.write("MatB*MatA: " + str(ans) + "\n")
            if ch1 == '3':
                ans = A * A
                with open("AnsMemory.txt", 'a') as fh:
                    fh.write("MatA*MatA: " + str(ans) + "\n")
            if ch1 == '4':
                ans = B * B
                with open("AnsMemory.txt", 'a') as fh:
                    fh.write("MatB*MatB: " + str(ans) + "\n")
            print(ans)
            break
        if ch == 'g':
            while True:
                ch1 = input("MatA or MatB? [A/B] ")
                ch1 = ch1.lower()
                if ch1 in 'ab':
                    break
            if ch1 == 'a':
                ans = sy.det(A)
                with open("AnsMemory.txt", 'a') as fh:
                    fh.write("detA: " + str(ans) + "\n")
            if ch1 == 'b':
                ans = sy.det(B)
                with open("AnsMemory.txt", 'a') as fh:
                    fh.write("detB: " + str(ans) + "\n")
            print(ans)
            break
        if ch == 'h':
            while True:
                ch1 = input("MatA or MatB? [A/B] ")
                ch1 = ch1.lower()
                if ch1 in 'ab':
                    break
            if ch1 == 'a':
                if sy.det(A) != 0:
                    ans = A.inv()
                    with open("AnsMemory.txt", 'a') as fh:
                        fh.write("inv(A): " + str(ans) + "\n")
                else:
                    print("DetA=0, Inverse doesn't exist.")
            if ch1 == 'b':
                if sy.det(B) != 0:
                    ans = B.inv()
                    with open("AnsMemory.txt", 'a') as fh:
                        fh.write("inv(B): " + str(ans) + "\n")

                else:
                    print("DetB=0, Inverse doesn't exist.")
            print(ans)
            break
        if ch == 'i':
            while True:
                ch1 = input("MatA or MatB? [A/B] ")
                ch1 = ch1.lower()
                if ch1 in 'ab':
                    break
            if ch1 == 'a':
                if sy.det(A) != 0:
                    ans = sy.det(A) * A.inv()
                    with open("AnsMemory.txt", 'a') as fh:
                        fh.write("Adj(A): " + str(ans) + "\n")
                else:
                    print("DetA=0, Inverse doesn't exist; Cannot calculate Adjoint.")
            if ch1 == 'b':
                if sy.det(B) != 0:
                    ans = sy.det(B) * B.inv()
                    with open("AnsMemory.txt", 'a') as fh:
                        fh.write("Adj(B): " + str(ans) + "\n")
                else:
                    print("DetB=0, Inverse doesn't exist; Cannot calculate Adjoint")
            print(ans)
            break

    if CONTINUE():
        MAT(A, B)
    else:
        MMenu()


def createMAT():
    mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    print("[E11 E12 E13]\n[E21 E22 E23]\n[E31 E32 E33]")
    try:
        for i in range(3):
            for j in range(3):
                print('E', i + 1, j + 1, '?', sep="", end="")
                mat[i][j] = eval(input())
        mat = sy.Matrix(mat)
    except:
        print("Enter valid values only")
        createMAT()
    return mat


# END -- 3. Matrices and Determinants -------


# 4. Vector Algebra -------------------------
def VEC(A=np.array([0, 0, 0]), B=np.array([0, 0, 0])):
    print("\n\n\n")
    while True:
        print("Choose Operation: ")
        print("a. Enter VectA")
        print("b. Enter VectB")
        print("c. Edit Vect")
        print("d. Add Vectors")
        print("e. Subtract Vectors")
        print("f. Dot Product")
        print("g. Cross Product")
        print("h. Magnitude")
        print("i. Angle between vectors")
        while True:
            ch = input("Enter valid choice [a-i]: ")
            ch = ch.lower()
            if ch in 'abcdefghi':
                break
        if ch == 'a':
            A = createVECT()
        if ch == 'b':
            B = createVECT()
        if ch == 'c':
            while True:
                ch1 = input("VectorA or VectorB? [A/B]")
                ch1 = ch.lower()
                if ch in 'ab':
                    break
            if ch1 == 'a':
                A = createVECT()
            if ch1 == 'b':
                B = createVECT()

        if ch == 'd':
            ans = A + B
            print(ans[0], "i + ", ans[1], "j + ", ans[2], "k", sep="")
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("VecA+VecB: " + str(ans) + "\n")
            break
        if ch == 'e':
            ans = A - B
            print(ans[0], "i + ", ans[1], "j + ", ans[2], "k", sep="")
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("VecA-VecB: " + str(ans) + "\n")
            break
        if ch == 'f':
            C = A * B
            ans = C[0] + C[1] + C[2]
            print("Dot product: ", ans)
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("VecA . VecB " + str(ans) + "\n")
            break
        if ch == 'g':
            ans = [(A[1] * B[2] - B[1] * A[2]), (B[0] * A[2] - A[0] * B[2]), (A[0] * B[1] - A[1] * B[0])]
            print("Cross Product: ", ans[0], "i + ", ans[1], "j + ", ans[2], "k", sep="")
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("VecA x VecB: " + str(ans) + "\n")
            break
        if ch == 'h':
            while True:
                ch1 = input("VectorA or VectorB? [A/B]")
                ch1 = ch1.lower()
                if ch1 in 'ab':
                    break

            if ch1 == 'a':
                C = np.array(A)
            if ch1 == 'b':
                C = np.array(B)
            C = C ** 2
            ans = sqrt(C[0] + C[1] + C[2])
            print(ans)
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("Vec_Magnitude: " + str(ans) + "\n")
            break
        if ch == 'i':
            cpyA = A ** 2
            cpyB = B ** 2
            C = A * B
            dp = C[0] + C[1] + C[2]
            MA = sqrt(cpyA[0] + cpyA[1] + cpyA[2])
            MB = sqrt(cpyB[0] + cpyB[1] + cpyB[2])
            ans = degrees(np.arccos(dp / (MA * MB)))
            print(ans)
            with open("AnsMemory.txt", 'a') as fh:
                fh.write("Angle b/w VecA, VecB: " + str(ans) + "\n")
            break

    if CONTINUE():
        VEC(A, B)
    else:
        MMenu()


def createVECT():
    vect = np.array([0, 0, 0])
    try:
        for i in range(3):
            if i == 0:
                vect[0] = float(input("i? "))
            if i == 1:
                vect[1] = float(input("j? "))
            if i == 2:
                vect[2] = float(input("k? "))
    except:
        print("Enter valid values only")
        createVECT()

    return vect


# END -- 4. Vector Algebra ---------------

# 5. Equation Solver ---------------------
def EQ():
    print('\n\n\n')
    while True:
        print("Select the type of equation:")
        print("a. ax+b=0")
        print("b. ax²+bx+c=0")
        print("c. ax³+bx²+cx+d=0")
        print("d. ax⁴+bx³+cx²+dx+e=0")
        while True:
            ch = input("Enter valid choice [a-d]: ")
            ch = ch.lower()
            if ch in 'abcd':
                break
        print('\n')
        try:
            if ch == 'a':
                a = int(input("Enter a: "))
                b = int(input("Enter b: "))
                x = sy.Symbol('x')
                exp = a * x + b
                res = sy.solve(exp)
                break
            if ch == 'b':
                a = int(input("Enter a: "))
                b = int(input("Enter b: "))
                c = int(input("Enter c: "))
                x = sy.Symbol('x')
                exp = a * x ** 2 + b * x + c
                res = sy.solve(exp)
                break
            if ch == 'c':
                a = int(input("Enter a: "))
                b = int(input("Enter b: "))
                c = int(input("Enter c: "))
                d = int(input("Enter d: "))
                x = sy.Symbol('x')
                exp = a * x ** 3 + b * x ** 2 + c * x + d
                res = sy.solve(exp)
                break
            if ch == 'd':
                a = int(input("Enter a: "))
                b = int(input("Enter b: "))
                c = int(input("Enter c: "))
                d = int(input("Enter d: "))
                e = int(input("Enter e: "))
                x = sy.Symbol('x')
                exp = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
                res = sy.solve(exp)
                break
        except:
            print("Enter numerical values only.")
            EQ()

    print('Ans:', res)
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("EQ: " + str(res) + "\n")

    if CONTINUE():
        EQ()
    else:
        MMenu()


# END -- 5. Equation Solver ------------

# 6. Inequality Solver -----------------
def IEQ():
    print('\n\n\n')
    while True:
        print("Select the type of inequality:")
        print("a. ax+b > 0")
        print("b. ax+b < 0")
        print("c. ax²+bx+c > 0")
        print("d. ax²+bx+c < 0")
        while True:
            ch = input("Enter valid choice [a-d]: ")
            ch = ch.lower()
            if ch in 'abcd':
                break
        print('\n')
        if ch == 'a':
            a = int(input("Enter a: "))
            b = int(input("Enter b: "))
            x = sy.Symbol('x')
            exp = a * x + b > 0
            res = sy.solve(exp)
            break
        if ch == 'b':
            a = int(input("Enter a: "))
            b = int(input("Enter b: "))
            x = sy.Symbol('x')
            exp = a * x + b < 0
            res = sy.solve(exp)
            break
        if ch == 'c':
            a = int(input("Enter a: "))
            b = int(input("Enter b: "))
            c = int(input("Enter c: "))
            x = sy.Symbol('x')
            exp = a * x ** 2 + b * x + c > 0
            res = sy.solve(exp)
            break
        if ch == 'd':
            a = int(input("Enter a: "))
            b = int(input("Enter b: "))
            c = int(input("Enter c: "))
            x = sy.Symbol('x')
            exp = a * x ** 2 + b * x + c < 0
            res = sy.solve(exp)
            break

    print('Ans:', res)
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("IEQ: " + str(res) + "\n")

    if CONTINUE():
        IEQ()
    else:
        MMenu()


# END -- 6. Inequality Solver ---------------

# 7. Grapher --------------------------------
def GRAPH():
    print('\n\n\n')
    print("Enter a polynomial equation in x: ")
    x = np.arange(-100, 100, 0.1)
    eq = input("Enter f(x): ")
    y = eval(eq)
    pl.plot(x, y)
    pl.ylabel("f(x)=" + eq)
    pl.xlabel("Values")
    pl.title("Graph of y=f(x)")
    pl.grid()
    pl.show()

    if CONTINUE():
        GRAPH()
    else:
        MMenu()


# END -- 7. Grapher -------------------------

# 8. Complex Numbers ------------------------
def COMP(Z1=0 + 0j, Z2=0 + 0j):
    print('\n\n')
    while True:
        print("\nSelect complex operations: ")
        print("NOTE: sqrt(-1) = j")
        print("a. Enter number 1: ")
        print("b. Enter number 2: ")
        print("c. Addition")
        print("d. Subtraction")
        print("e. Multiplication")
        print("f. Division")
        print("g. Square Root")
        while True:
            ch = input("Enter valid choice [a-g]: ")
            ch = ch.lower()
            if ch in 'abcdefg':
                break
        try:

            if ch == 'a':
                Z1 = complex(input("Enter number 1: "))
            if ch == 'b':
                Z2 = complex(input("Enter number 2: "))
            if ch == 'c':
                res = Z1 + Z2
                with open("AnsMemory.txt", 'a') as fh:
                    fh.write("Z1+Z2: " + str(res) + "\n")
                break
            if ch == 'd':
                print("\n")
                print("1. Z1-Z2")
                print("2. Z2-Z1")
                while True:
                    ch1 = input("Enter valid choice [1/2]: ")
                    if ch1 in '12':
                        break
                    if ch1 == '1':
                        res = Z1 - Z2
                        with open("AnsMemory.txt", 'a') as fh:
                            fh.write("Z1-Z2: " + str(res) + "\n")
                    if ch1 == '2':
                        res = Z2 - Z1
                        with open("AnsMemory.txt", 'a') as fh:
                            fh.write("Z2-Z1: " + str(res) + "\n")
                break
            if ch == 'e':
                res = Z1 * Z2
                with open("AnsMemory.txt", 'a') as fh:
                    fh.write("Z1*Z2: " + str(res) + "\n")
                break
            if ch == 'f':
                res = Z1 / Z2
                with open("AnsMemory.txt", 'a') as fh:
                    fh.write("Z1/Z2: " + str(res) + "\n")
                break
            if ch == 'g':
                print("\n")
                print("1. Z1?")
                print("2. Z2?")
                while True:
                    ch1 = input("Enter valid choice [1/2]: ")
                    if ch1 in '12':
                        break
                if ch1 == '1':
                    res = cqrt(Z1)
                    with open("AnsMemory.txt", 'a') as fh:
                        fh.write("sqrtZ1: " + str(res) + "\n")
                if ch1 == '2':
                    res = cqrt(Z2)
                    with open("AnsMemory.txt", 'a') as fh:
                        fh.write("sqrtZ2: " + str(res) + "\n")
                break
        except:
            print("Enter valid values only")
            COMP()


    print(res)

    if CONTINUE():
        COMP(Z1, Z2)
    else:
        MMenu()


# END -- 8. Complex Numbers -----------------

# 9. Conversions ----------------------------
def CONV():
    print('\n\n\n')
    while True:
        print("Select the conversion: ")
        print("a. Length")
        print("b. Temperature")
        print("c. Area")
        print("d. Volume")
        print("e. Mass/Weight")
        while True:
            ch = input("Enter valid choice [a-e]: ")
            ch = ch.lower()
            if ch in 'abcde':
                break
        if ch == 'a':
            res = len_CONV()
            break
        if ch == 'b':
            res = temp_CONV()
            break
        if ch == 'c':
            res = area_CONV()
            break
        if ch == 'd':
            res = vol_CONV()
            break
        if ch == 'e':
            res = mass_CONV()
            break
    print(res)
    with open("AnsMemory.txt", 'a') as fh:
        fh.write("CONV: " + str(res) + "\n")

    if CONTINUE():
        CONV()
    else:
        MMenu()


def len_CONV():
    print('\n')
    while True:
        print("Select: ")
        print("a. Feet to Meter")
        print("b. Yard to Meter")
        print("c. Mile to Kilometer")
        print("d. Inch to Centimeter")
        while True:
            ch = input("Enter valid choice [a-d]: ")
            ch = ch.lower()
            if ch in 'abcd':
                break
        try:
            if ch == 'a':
                u_in = float(input("Enter value in Feet: "))
                u_out = str(round(u_in / 3.281, 3)) + ' m'
                break
            if ch == 'b':
                u_in = float(input("Enter value in Yard: "))
                u_out = str(round(u_in / 1.094, 3)) + ' m'
                break
            if ch == 'c':
                u_in = float(input("Enter value in Mile: "))
                u_out = str(round(u_in * 1.60935, 3)) + ' km'
                break
            if ch == 'd':
                u_in = float(input("Enter value in Inch: "))
                u_out = str(round(u_in * 2.54, 3)) + ' cm'
                break
        except:
            print("Enter valid input only")
            len_CONV()

    return u_out


def temp_CONV():
    print('\n')
    while True:
        print("Select: ")
        print("a. Celsius to Fahrenheit")
        print("b. Celsius to Kelvin")
        print("c. Fahrenheit to Kelvin")
        while True:
            ch = input("Enter valid choice [a-c]: ")
            ch = ch.lower()
            if ch in 'abc':
                break
        try:
            if ch == 'a':
                u_in = float(input("Enter value in Celsius: "))
                u_out = str(round((u_in * (9 / 5)) + 32, 3)) + '°F'
                break
            if ch == 'b':
                u_in = float(input("Enter value in Celsius: "))
                u_out = str(round(u_in + 273.15, 3)) + 'K'
                break
            if ch == 'c':
                u_in = float(input("Enter value in Fahrenheit: "))
                u_out = str(round((u_in - 32) * (5 / 9) + 273.15, 3)) + 'K'
                break
        except:
            print("Enter valid values only")
            temp_CONV()

    return u_out


def area_CONV():
    print('\n')
    while True:
        print("Select: ")
        print("a. Sq.Kilometer to Sq.Meter")
        print("b. Hectare to Sq.Kilometer")
        print("c. Sq.Foot to Sq.Meter")
        print("d. Acre to Sq.Meter")
        print("e. Acre to Hectare")
        while True:
            ch = input("Enter valid choice [a-e]: ")
            ch = ch.lower()
            if ch in 'abcde':
                break
        try:
            if ch == 'a':
                u_in = float(input("Enter value in sq. Kilometer: "))
                u_out = str(round(u_in * 10 ** 6, 3)) + ' m²'
                break
            if ch == 'b':
                u_in = float(input("Enter value in Hectare: "))
                u_out = str(round(u_in / 100, 3)) + ' km²'
                break
            if ch == 'c':
                u_in = float(input("Enter value in sq. Foot: "))
                u_out = str(round(u_in / 10.764, 3)) + ' m²'
                break
            if ch == 'd':
                u_in = float(input("Enter value in Acre: "))
                u_out = str(round(u_in * 4046.856, 3)) + ' m²'
                break
            if ch == 'e':
                u_in = float(input("Enter value in Acre: "))
                u_out = str(round(u_in / 2.471, 3)) + ' ha'
                break
        except:
            print("Enter valid values only")
            area_CONV()

    return u_out


def vol_CONV():
    print('\n')
    while True:
        print("Select: ")
        print("a. Liter to Cubic Meter")
        print("b. Gallon to Liter")
        print("c. Cup to Milliliter")
        while True:
            ch = input("Enter valid choice [a-c]: ")
            ch = ch.lower()
            if ch in 'abc':
                break
        try:
            if ch == 'a':
                u_in = float(input("Enter value in Liter: "))
                u_out = str(round(u_in / 1000, 3)) + ' m³'
                break
            if ch == 'b':
                u_in = float(input("Enter value in Gallon: "))
                u_out = str(round(u_in * 3.785, 3)) + ' L'
                break
            if ch == 'c':
                u_in = float(input("Enter value in Cup: "))
                u_out = str(round(u_in * 236.588, 3)) + ' mL'
                break
        except:
            print("Enter numerical values only")
            vol_CONV()
    return u_out


def mass_CONV():
    print('\n')
    while True:
        print("Select: ")
        print("a. Pound to Kilogram")
        print("b. Ounce to Gram")
        print("c. Carat to Milligram")
        print("d. Atomic Mass Unit to Gram")
        while True:
            ch = input("Enter valid choice [a-d]: ")
            ch = ch.lower()
            if ch in 'abcd':
                break
        try:
            if ch == 'a':
                u_in = float(input("Enter value in Pound: "))
                u_out = str(round(u_in * 2.205, 3)) + ' kg'
                break
            if ch == 'b':
                u_in = float(input("Enter value in Ounce: "))
                u_out = str(round(u_in * 28.35, 3)) + ' g'
                break
            if ch == 'c':
                u_in = float(input("Enter value in Carat: "))
                u_out = str(round(u_in * 200, 3)) + ' mg'
                break
            if ch == 'd':
                u_in = float(input("Enter value in amu: "))
                u_out = f"{Decimal(str(u_in / (6.0221409 * 10 ** 23))):.2E}"
                break
        except:
            print("Enter valid numerical values only")
            mass_CONV()

    return u_out


# END -- 9. Conversions ---------------

# 10. Answer Memory -------------------
def AnsMem():
    print('\n')
    try:
        file_handle = open("AnsMemory.txt", "r")
        ans_mem = file_handle.readlines()[::-1]
        file_handle.close()

        try:
            n = int(input("Enter the number of answers to retrieve from answer memory: "))
        except:
            print("Enter numerical values only")
            AnsMem()

        try:
            for i in range(n):
                print(ans_mem[i].strip())
        except:
            pass

        if CONTINUE():
            AnsMem()
        else:
            MMenu()
    except:
        print("\nAnswer Memory empty.")
        if CONTINUE():
            MMenu()
        else:
            EXIT()


# END -- 10. Answer Memory ------------

# MainMenu
def MMenu():
    ch = -1
    print("Choose Operation: ")
    print("01. Basic Arithmetic Operations")
    print("02. Trigonometric Functions")
    print("03. Matrices and Determinants")
    print("04. Vector Algebra")
    print("05. Equations")
    print("06. Inequalities")
    print("07. Grapher")
    print("08. Complex")
    print("09. Conversions")
    print("10. Answer Memory")
    print("11. Clear AnsMemory")
    print("0. EXIT")

    # Accept valid input within bounds
    while ch not in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'):
        ch = (input("Enter valid value [0-11]: "))

    # Execute proper operation
    if ch == '1': BAO()
    if ch == '2': TF()
    if ch == '3': MAT()
    if ch == '4': VEC()
    if ch == '5': EQ()
    if ch == '6': IEQ()
    if ch == '7': GRAPH()
    if ch == '8': COMP()
    if ch == '9': CONV()
    if ch == '10': AnsMem()
    if ch == '11':
        if os.path.exists("AnsMemory.txt"):
            os.remove("AnsMemory.txt")
            print("Memory cleared. \n\n")
        else:
            print("Answer Memory empty.\n\n")
        if CONTINUE():
            MMenu()
        else:
            EXIT()
    if ch == '0': EXIT()



print("*******************************")
print("|                             |")
print("|     Numerical Analyst       |")
print("|                             |")
print("*******************************")
print("\n\n")
MMenu()
