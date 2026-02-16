import re
from typing import Dict, List, Any, Optional, Tuple
from indexing.schema import Program, ProgramStep
import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for timeout"""
    def signal_handler(signum, frame):
        raise TimeoutException("Execution timed out")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class ProgramExecutor:
    """Execute FinQA DSL programs"""
    
    OPERATIONS = {
        'add': lambda a, b: a + b,
        'subtract': lambda a, b: a - b,
        'multiply': lambda a, b: a * b,
        'divide': lambda a, b: a / b if b != 0 else float('inf'),
        'percent': lambda a, b: (a / b * 100) if b != 0 else float('inf'),
        'diff': lambda a, b: abs(a - b),
        'max': lambda *args: max(args[0]) if isinstance(args[0], list) else max(args),
        'min': lambda *args: min(args[0]) if isinstance(args[0], list) else min(args),
        'sum': lambda *args: sum(args[0]) if isinstance(args[0], list) else sum(args),
        'avg': lambda *args: sum(args[0]) / len(args[0]) if isinstance(args[0], list) and len(args[0]) > 0 else 0
    }
    
    def __init__(self, timeout_seconds: int = 5):
        self.timeout_seconds = timeout_seconds
        self.variables: Dict[str, float] = {}
        self.steps: List[ProgramStep] = []
    
    def parse_program(self, program_text: str) -> List[str]:
        """Parse program text into executable lines"""
        lines = []
        for line in program_text.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('REASONING') and not line.startswith('PROGRAM'):
                lines.append(line)
        return lines
    
    def execute(self, program_text: str) -> Tuple[Optional[float], Program, Optional[str]]:
        """Execute a program and return result
        
        Returns:
            (final_answer, program_object, error_message)
        """
        self.variables = {}
        self.steps = []
        
        try:
            with time_limit(self.timeout_seconds):
                lines = self.parse_program(program_text)
                
                for line in lines:
                    try:
                        self._execute_line(line)
                    except Exception as e:
                        error_msg = f"Error executing line '{line}': {str(e)}"
                        program = Program(steps=self.steps, variables=self.variables)
                        return None, program, error_msg
                
                # Get final answer
                final_answer = None
                if 'answer' in self.variables:
                    final_answer = self.variables['answer']
                elif self.steps:
                    # Use last computed result
                    final_answer = self.steps[-1].result
                
                program = Program(
                    steps=self.steps,
                    final_answer=final_answer,
                    variables=self.variables
                )
                
                return final_answer, program, None
                
        except TimeoutException:
            program = Program(steps=self.steps, variables=self.variables)
            return None, program, "Execution timeout"
        except Exception as e:
            program = Program(steps=self.steps, variables=self.variables)
            return None, program, f"Execution error: {str(e)}"
    
    def _execute_line(self, line: str):
        """Execute a single program line"""
        # Parse: variable = operation(args)
        if '=' not in line:
            return
        
        var_name, expression = line.split('=', 1)
        var_name = var_name.strip()
        expression = expression.strip()
        
        # Parse operation and arguments
        match = re.match(r'(\w+)\((.*)\)', expression)
        if not match:
            # Direct assignment: const_0 = 5.2
            try:
                value = float(expression)
                self.variables[var_name] = value
                step = ProgramStep(
                    operation='const',
                    arguments=[value],
                    result=value
                )
                self.steps.append(step)
                return
            except ValueError:
                raise ValueError(f"Invalid expression: {expression}")
        
        operation = match.group(1)
        args_str = match.group(2)
        
        if operation not in self.OPERATIONS:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Parse arguments
        args = self._parse_arguments(args_str)
        
        # Execute operation
        func = self.OPERATIONS[operation]
        result = func(*args)
        
        self.variables[var_name] = result
        
        step = ProgramStep(
            operation=operation,
            arguments=args,
            result=result
        )
        self.steps.append(step)
    
    def _parse_arguments(self, args_str: str) -> List[Any]:
        """Parse function arguments"""
        args = []
        
        # Handle list arguments: [a, b, c]
        if args_str.strip().startswith('['):
            list_match = re.match(r'\[(.*)\]', args_str.strip())
            if list_match:
                items = list_match.group(1).split(',')
                list_args = []
                for item in items:
                    item = item.strip()
                    if item in self.variables:
                        list_args.append(self.variables[item])
                    else:
                        try:
                            list_args.append(float(item))
                        except ValueError:
                            raise ValueError(f"Invalid list item: {item}")
                return [list_args]
        
        # Handle regular arguments
        for arg in args_str.split(','):
            arg = arg.strip()
            
            # Check if it's a variable
            if arg in self.variables:
                args.append(self.variables[arg])
            else:
                # Try to parse as number
                try:
                    args.append(float(arg))
                except ValueError:
                    raise ValueError(f"Invalid argument: {arg}")
        
        return args
    
    def verify_program(self, program_text: str, available_numbers: List[float]) -> Tuple[bool, str]:
        """Verify that program only uses available numbers"""
        # Extract all constant numbers from program
        lines = self.parse_program(program_text)
        used_numbers = []
        
        for line in lines:
            # Find all numeric constants
            numbers = re.findall(r'-?\d+\.?\d*', line)
            used_numbers.extend([float(n) for n in numbers])
        
        # Check if all used numbers are available
        tolerance = 1e-6
        for num in used_numbers:
            found = False
            for avail in available_numbers:
                if abs(num - avail) < tolerance:
                    found = True
                    break
            
            if not found:
                return False, f"Number {num} not found in evidence"
        
        return True, "All numbers verified"
    
    def extract_reasoning_and_program(self, llm_output: str) -> Tuple[str, str]:
        """Extract reasoning and program from LLM output"""
        reasoning = ""
        program = ""
        
        # Split by markers
        if "REASONING:" in llm_output and "PROGRAM:" in llm_output:
            parts = llm_output.split("PROGRAM:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            program = parts[1].strip()
        elif "PROGRAM:" in llm_output:
            program = llm_output.split("PROGRAM:")[1].strip()
        else:
            # Assume entire output is program
            program = llm_output.strip()
        
        return reasoning, program


if __name__ == "__main__":
    executor = ProgramExecutor()
    
    # Test program
    test_program = """
    const_0 = 100
    const_1 = 50
    result_0 = add(const_0, const_1)
    const_2 = 2
    result_1 = divide(result_0, const_2)
    answer = result_1
    """
    
    final_answer, program, error = executor.execute(test_program)
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"Final answer: {final_answer}")
        print(f"Steps executed: {len(program.steps)}")
        print(f"Variables: {program.variables}")
