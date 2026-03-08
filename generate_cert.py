"""
SSL Certificate Generator for Visual Navigation Assistant
Generates self-signed certificates for HTTPS support
"""

from OpenSSL import crypto
import os

def generate_self_signed_cert():
    """Generate self-signed SSL certificate and key"""
    
    print("🔐 Generating SSL Certificate...")
    
    # Generate private key
    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)
    
    # Generate certificate
    cert = crypto.X509()
    cert.get_subject().C = "US"
    cert.get_subject().ST = "State"
    cert.get_subject().L = "City"
    cert.get_subject().O = "University Project"
    cert.get_subject().OU = "Visual Navigation Assistant"
    cert.get_subject().CN = "localhost"
    
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for 1 year
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    cert.sign(key, 'sha256')
    
    # Save certificate
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    print("✅ Created: cert.pem")
    
    # Save private key
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
    print("✅ Created: key.pem")
    
    print("\n🎉 SSL certificates generated successfully!")
    print("📁 Files created in current directory:")
    print("   - cert.pem (certificate)")
    print("   - key.pem (private key)")
    print("\n🚀 You can now run your Flask app with HTTPS!")
    print("   Run: python app.py")

if __name__ == "__main__":
    # Check if certificates already exist
    if os.path.exists("cert.pem") or os.path.exists("key.pem"):
        response = input("⚠️  Certificates already exist. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("❌ Certificate generation cancelled.")
            exit()
    
    try:
        generate_self_signed_cert()
    except Exception as e:
        print(f"\n❌ Error generating certificates: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure pyopenssl is installed: pip install pyopenssl")
        print("2. Check you have write permissions in current directory")
        print("3. Close any files that might be in use")