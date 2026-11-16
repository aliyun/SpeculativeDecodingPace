from ipaddress import ip_address
import re
from urllib.parse import urlsplit


def http_url(value, *, origin=False, allow_query=False):
    if (not isinstance(value, str) or not value or '\\' in value
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError('invalid HTTP URL')
    try:
        parsed = urlsplit(value)
        host, port = parsed.hostname, parsed.port
    except ValueError as exc:
        raise ValueError('invalid HTTP URL') from exc
    if (parsed.scheme != 'http' or not host or parsed.username is not None
            or parsed.password is not None or '#' in value
            or (not allow_query and '?' in value)
            or parsed.netloc.endswith(':')
            or (port is not None and not 1 <= port <= 65535)):
        raise ValueError('HTTP URL requires a host and no credentials or fragment')
    try:
        address = ip_address(host)
    except ValueError:
        try:
            hostname = host.encode('idna').decode('ascii').rstrip('.')
        except UnicodeError as exc:
            raise ValueError('invalid HTTP hostname') from exc
        if len(hostname) > 253 or any(not re.fullmatch(
                r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', label)
                for label in hostname.split('.')):
            raise ValueError('invalid HTTP hostname')
    else:
        if address.is_unspecified or address.is_multicast:
            raise ValueError('HTTP URL requires a destination address')
        if address.version == 6 and not re.fullmatch(
                r'\[[0-9A-Fa-f:.]+\](?::[0-9]+)?', parsed.netloc):
            raise ValueError('invalid IPv6 HTTP authority')
    if origin and (parsed.path not in ('', '/') or '?' in value):
        raise ValueError('edge URL must be an HTTP origin without a path or query')
    return parsed


def loopback_host(host):
    if host.rstrip('.').lower() == 'localhost':
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, 'ipv4_mapped', None)
    return address.is_loopback or (mapped is not None and mapped.is_loopback)
