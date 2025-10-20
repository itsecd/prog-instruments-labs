PATH_TRAVEL_PATTERNS = [
     r'\.\./',
    r'/etc/passwd',
    r'/etc/shadow', 
    r'/proc/self',
    r'\.\.\\'
]

XSS_PATTERNS = [
    r'<script[^>]*>.*?</script>',
    r'javascript:\s*\w+', 
    r'on(load|error|click|mouse)\s*=',
    r'<iframe[^>]*src=',
    r'<svg[^>]*on\w+=',
    r'alert\s*\([^)]*\)'
]

SQL_INJECTION_PATTERNS = [
    r"'\s*(union|select|insert|update|delete|drop|create|alter)\s",
    r"(union|select|insert|update|delete|drop|create|alter)\s.*'",
    r"'\s*;\s*(select|insert|update|delete|drop)",
    r"(union\s+select|select\s+from|insert\s+into|update\s+set|delete\s+from)",
    r"or\s+['\"]?\s*1\s*=\s*1\s*['\"]?",
    r"'\s*--\s*$",
    r";\s*--\s*$",
    r"union\s+select\s+null",
    r"waitfor\s+delay\s+'"
]

MALICIOUS_AGENT_PATTERNS = {
    'sqlmap': r'sqlmap',
    'nikto': r'nikto',
    'nmap': r'nmap',
    'metasploit': r'metasploit',
    'burp': r'burp',
    'zap': r'owasp.zap',
    'scanner': r'scanner|scanning',
}