import math


class expression:
    """Base expression class. Represents a singular variable, like x."""

    def __init__(self, name):
        """Init function.
        Args:
            Name: name of the variable, like "x."
        """
        self.name = name

    def eval(self, values):
        """Evaluation function, returns value of expression/variable given value parameters.

        Args:
            values (dict): a dicitonary of variable names:values.

        Returns:
            float representing value of variable at given evaluation.
        """
        return values[self.name]

    def diff(self, values, diffto):
        """Differentiate variable / expression with the current values.

        Args:
            values (dict): a dicitonary of variable names:values.
            diffto (string): variable with respect to which differentiation happens.

        Returns:
            1 or 0 -- depends if this is the variable we're differentiating with respect to.
        """
        return 1 if diffto == self.name else 0

    # the following are all overrides: (all other subsequent classes inherit)
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return addition(self, other)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return addition(other, self)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return subtraction(self, other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return subtraction(other, self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return multiplication(self, other)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return multiplication(other, self)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return division(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return division(other, self)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return exponent(self, other)

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return exponent(other, self)


class constant(expression):
    """Constant expression class. Represents a constant value, like 2 or 3.14."""

    def __init__(self, value):
        """Init function.
        Args:
            value: numeric value of the constant.
        """
        self.value = value

    def eval(self, values):
        """Evaluation function, returns the constant value.

        Args:
            values (dict): a dicitonary of variable names:values (not used for constants).

        Returns:
            float representing the constant value.
        """
        return self.value

    def diff(self, values, diffto):
        """Differentiate constant with respect to any variable.

        Args:
            values (dict): a dicitonary of variable names:values (not used).
            diffto (string): variable with respect to which differentiation happens (not used).

        Returns:
            0 -- derivative of a constant is always 0.
        """
        return 0


class addition(expression):
    def __init__(self, first, second):
        """Addition of two expressions.

        Args:
            first (expression): an expression
            second (expression): another expression
        """
        self.first = first
        self.second = second

    def eval(self, values):
        return self.first.eval(values) + self.second.eval(values)

    def diff(self, values, diffto):
        return self.first.diff(values, diffto) + self.second.diff(values, diffto)


class subtraction(expression):
    def __init__(self, first, second):
        """Subtraction of two expressions.

        Args:
            first (expression): an expression
            second (expression): another expression
        """
        self.first = first
        self.second = second

    def eval(self, values):
        return self.first.eval(values) - self.second.eval(values)

    def diff(self, values, diffto):
        return self.first.diff(values, diffto) - self.second.diff(values, diffto)


class multiplication(expression):
    def __init__(self, first, second):
        """Multiplication of two expressions.

        Args:
            first (expression): an expression
            second (expression): another expression
        """
        self.first = first
        self.second = second

    def eval(self, values):
        return self.first.eval(values) * self.second.eval(values)

    def diff(self, values, diffto):
        # product rule, u'(x) * v(x) + u(x) * v'(x)
        return self.first.diff(values, diffto) * self.second.eval(
            values
        ) + self.first.eval(values) * self.second.diff(values, diffto)


class division(expression):
    def __init__(self, first, second):
        """Division of two expressions.

        Args:
            first (expression): an expression
            second (expression): another expression
        """
        self.first = first
        self.second = second

    def eval(self, values):
        return self.first.eval(values) / self.second.eval(values)

    def diff(self, values, diffto):
        # quotient rule: (v(x)u'(x) - v'(x)u(x)) / (v(x) ^ 2)
        return (
            self.second.eval(values) * self.first.diff(values, diffto)
            - self.second.diff(values, diffto) * self.first.eval(values)
        ) / (self.second.eval(values) ** 2)


class exponent(expression):
    def __init__(self, first, second):
        """Exponent of two expressions.

        Args:
            first (expression): an expression
            second (expression): another expression
        """
        self.first = first
        self.second = second

    def eval(self, values):
        return self.first.eval(values) ** self.second.eval(values)

    def diff(self, values, diffto):
        # power rule -- generalized
        return self.eval(values) * (natlog(self.first) * self.second).diff(
            values, diffto
        )


### Functions -- single input###
# These MUST account for the chain rule.
# They also aren't called via overrides, meaning that they should handle casts to constant if need be.


# class linear_activation
# An example class of what you'll be implementing
class linear(expression):
    def __init__(self, first):
        """Linear activation function.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first

    def eval(self, values):
        return self.first.eval(values)

    def diff(self, values, diffto):
        # chain rule
        return self.first.diff(values, diffto)


class natlog(expression):
    def __init__(self, first):
        """Natural log of an expression.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first

    def eval(self, values):
        return math.log(self.first.eval(values))

    def diff(self, values, diffto):
        # chain rule
        return 1 / self.first.eval(values) * self.first.diff(values, diffto)


class step:
    def __init__(self, first):
        """Step activation function.

        Args:
            first (expressions): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first

    def eval(self, values):
        # TODO: Implement eval()
        return 0

    def diff(self, values, diffto):
        # TODO: Implement diff()
        return 0


# class sine
class sine(expression):
    def __init__(self, first):
        """sine of an expression.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first

    def eval(self, values):
        # TODO: Implement eval()
        return 0

    def diff(self, values, diffto):
        # chain rule
        # TODO: Implement diff()
        return 0


# class cosine
class cosine(expression):
    def __init__(self, first):
        """cosine of an expression.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first

    def eval(self, values):
        # TODO: Implement eval()
        return 0

    def diff(self, values, diffto):
        # chain rule
        # TODO: Implement diff()
        return 0


# class tangent
class tangent(expression):
    def __init__(self, first):
        """tangent of an expression.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first

    def eval(self, values):
        # TODO: Implement eval()
        return 0

    def diff(self, values, diffto):
        # chain rule
        # TODO: Implement diff()
        return 0


# class tanh (tanh)
# We'll be giving you the diff() code for tanh
class hyperbolic_tan(expression):
    def __init__(self, first):
        """tanh. activation function

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first

    def eval(self, values):
        # TODO: Implement eval()
        return 0

    def diff(self, values, diffto):
        return (
            1
            / math.cosh(self.first.eval(values)) ** 2
            * self.first.diff(values, diffto)
        )


# class sigmoid
class sigmoid(expression):
    def __init__(self, first):
        """Sigmoid activation function.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first

    def eval(self, values):
        # TODO: Implement eval()
        return 0

    def diff(self, values, diffto):
        # chain rule
        # TODO: Implement diff()
        return 0


# class relu
class relu(expression):
    def __init__(self, first):
        """Rectified linear unit activation function.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first

    def eval(self, values):
        # TODO: Implement eval()
        return 0

    def diff(self, values, diffto):
        # chain rule
        # TODO: Implement diff()
        return 0


# class relu_leaky
class leaky_relu(expression):
    def __init__(self, first):
        """Leaky rectified linear unit activation function.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first

    def eval(self, values):
        # TODO: Implement eval()
        return 0

    def diff(self, values, diffto):
        # chain rule
        # TODO: Implement diff()
        return 0


# BONUS: Swish
class swish:
    def __init__(self, first):
        """Swish activation function.

        Args:
            first (expressions): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first

    def eval(self, values):
        # TODO: Implement eval()
        return 0

    def diff(self, values, diffto):
        # chain rule
        #  TODO: Implement diff()
        return 0
