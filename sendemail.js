import nodemailer from "nodemailer";

// 🔧 Configure transporter (use your email service)
const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: "tarang.termoid@gmail.com",        // 👈 your email
    pass: "wukn ewxl buhj pjii",           // 👈 NOT your real password (use App Password)
  },
});

// 🎨 HTML Email Template
const htmlTemplate = `
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a; padding:20px; font-family:Arial, sans-serif;">
  <tr>
    <td align="center">
      
      <table width="600" cellpadding="0" cellspacing="0" style="background:#111827; border-radius:10px; overflow:hidden; border:1px solid #1f2937;">
        
        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#4f46e5,#6366f1); color:#ffffff; padding:30px; text-align:center;">
            <h2 style="margin:0;">Tarang Portal Update</h2>
            <p style="margin-top:10px; opacity:0.9;">Now supports larger uploads 🚀</p>
          </td>
        </tr>

        <!-- Content -->
        <tr>
          <td style="padding:30px; color:#e5e7eb;">
            <p style="margin:0 0 10px;">Dear User,</p>

            <p style="line-height:1.6;">
              We’re excited to inform you that the Tarang Portal now supports
              <strong style="color:#c7d2fe;">larger document uploads</strong>.
            </p>

            <p style="line-height:1.6;">
              You can now upload bigger files seamlessly without previous limitations,
              ensuring a smoother and more efficient experience.
            </p>

            <!-- Highlight Box -->
            <table width="100%" style="margin:20px 0; background:#1e293b; border-left:4px solid #6366f1;">
              <tr>
                <td style="padding:15px; color:#cbd5f5;">
                  🚀 Upload larger files<br/>
                  ⚡ Faster performance<br/>
                  📁 Improved reliability
                </td>
              </tr>
            </table>

            <p style="line-height:1.6;">
              If you have any feedback or face any issues, feel free to reach out.
            </p>

            <p style="margin-top:20px;">
              Regards,<br/>
              <strong style="color:#ffffff;">Tarang Team</strong>
            </p>
          </td>
        </tr>

      </table>

      <!-- Footer -->
      <p style="color:#64748b; font-size:12px; margin-top:15px;">
        © Tarang • All rights reserved
      </p>

    </td>
  </tr>
</table>
`;

// 🚀 Send Email Function
async function sendEmail() {
  try {
    const info = await transporter.sendMail({
      from: '"Tarang Team" <tarang.termoid@gmail.com>',
      to: "jhaalok543@gmail.com", // 👈 change this
      subject: "🌊 Tarang Update: Upload Larger Documents",
      html: htmlTemplate, // ✅ THIS is the important part
    });

    console.log("Email sent:", info.response);
  } catch (error) {
    console.error("Error sending email:", error);
  }
}

// ▶️ Call function
sendEmail();