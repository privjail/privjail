"""dataserver.py -- brings up the privjail server-side process, listening
for privjail clients on the given port. Launched by gateway.py once the
encrypted directory is mounted.
"""
import sys
import privjail as pj

def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    pj.serve(port)

if __name__ == "__main__":
    main()
