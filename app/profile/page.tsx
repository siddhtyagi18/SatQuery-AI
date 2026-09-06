'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/authContext';
import {
  User as UserIcon,
  Mail,
  Calendar,
  Building,
  Shield,
  KeyRound,
  LogOut,
  ArrowRight,
  CheckCircle2,
  Lock,
  Eye,
  EyeOff,
  Save,
} from 'lucide-react';
import { toast } from 'sonner';

export default function ProfilePage() {
  const router = useRouter();
  const { user, updateProfile, logout } = useAuth();

  const [name, setName] = useState(user?.name || 'Operator');
  const [dob, setDob] = useState(user?.dob || '1995-08-15');
  const [organization, setOrganization] = useState(
    user?.organization || 'NRSC / Space Applications Centre (ISRO)'
  );

  // Password change form state
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [savingPassword, setSavingPassword] = useState(false);
  const [profileSaved, setProfileSaved] = useState(false);

  const handleSaveProfile = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      toast.error('Please provide a valid operator name');
      return;
    }
    updateProfile({
      name: name.trim(),
      dob,
      organization: organization.trim(),
    });
    setProfileSaved(true);
    toast.success('Operator profile details updated successfully');
    setTimeout(() => setProfileSaved(false), 3000);
  };

  const handleChangePassword = (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentPassword) {
      toast.error('Please enter your current clearance password');
      return;
    }
    if (newPassword.length < 8) {
      toast.error('New password must be at least 8 characters long');
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.error('New passwords do not match');
      return;
    }

    setSavingPassword(true);
    setTimeout(() => {
      setSavingPassword(false);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      toast.success('Clearance password updated successfully for your active session');
    }, 700);
  };

  return (
    <div className="page-shell max-w-5xl mx-auto pb-16 animate-fade-in-up flex flex-col gap-8">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-[var(--border-hairline)]">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="badge badge-cyan">OPERATOR PROFILE</span>
            <span className="badge badge-green flex items-center gap-1">
              <span className="relative flex h-1.5 w-1.5">
                <span className="animate-pulse-dot absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
              </span>
              ACTIVE SESSION
            </span>
          </div>
          <h1 className="text-display-lg text-[var(--text-primary)]">
            Mission Controller Account
          </h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            Manage your ground-station profile, clearance details, and security credentials.
          </p>
        </div>

        <Link
          href="/"
          className="btn-primary inline-flex items-center gap-2 self-start sm:self-center shadow-lg"
          style={{ padding: '10px 20px', fontSize: '0.82rem' }}
        >
          <span>Proceed to Dashboard</span>
          <ArrowRight className="w-4 h-4 arrow-slide" />
        </Link>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Operator Profile Info */}
        <div className="lg:col-span-2 flex flex-col gap-8">
          {/* Profile Details Card */}
          <div
            className="p-6 rounded-lg border border-[var(--border-hairline)] relative overflow-hidden"
            style={{
              background: 'var(--surface-1)',
              boxShadow: '0 4px 20px -2px rgba(0,0,0,0.3)',
            }}
          >
            <div className="flex items-center gap-3 mb-6 pb-4 border-b border-[var(--border-hairline)]">
              <div
                className="w-10 h-10 rounded-md flex items-center justify-center"
                style={{
                  background: 'color-mix(in srgb, var(--cyan) 14%, transparent)',
                  border: '1px solid color-mix(in srgb, var(--cyan) 30%, transparent)',
                }}
              >
                <UserIcon className="w-5 h-5 text-[var(--cyan)]" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-[var(--text-primary)]">
                  Personal Information
                </h2>
                <p className="text-xs text-[var(--text-muted)]">
                  Your identity as logged in the ISRO ground-station registry.
                </p>
              </div>
            </div>

            <form onSubmit={handleSaveProfile} className="flex flex-col gap-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                {/* Full Name */}
                <div>
                  <label className="hud-label block mb-2">Operator Full Name</label>
                  <div className="relative">
                    <UserIcon className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] pointer-events-none" />
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="e.g. Controller Sharma"
                      className="w-full pl-10 pr-4 py-2.5 rounded text-sm outline-none transition-all"
                      style={{
                        background: 'var(--surface-2)',
                        border: '1px solid var(--border-hairline)',
                        color: 'var(--text-primary)',
                      }}
                    />
                  </div>
                </div>

                {/* Date of Birth */}
                <div>
                  <label className="hud-label block mb-2">Date of Birth</label>
                  <div className="relative">
                    <Calendar className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] pointer-events-none" />
                    <input
                      type="date"
                      value={dob}
                      onChange={(e) => setDob(e.target.value)}
                      className="w-full pl-10 pr-4 py-2.5 rounded text-sm outline-none transition-all"
                      style={{
                        background: 'var(--surface-2)',
                        border: '1px solid var(--border-hairline)',
                        color: 'var(--text-primary)',
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Email Address (Read-only) */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="hud-label">Clearance Email (Ground ID)</label>
                  <span className="text-[0.62rem] font-mono text-[var(--green)] flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> VERIFIED SSO
                  </span>
                </div>
                <div className="relative">
                  <Mail className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] pointer-events-none" />
                  <input
                    type="email"
                    value={user?.email || 'controller@isro.gov.in'}
                    disabled
                    className="w-full pl-10 pr-4 py-2.5 rounded text-sm outline-none opacity-80 cursor-not-allowed font-mono"
                    style={{
                      background: 'var(--surface-0)',
                      border: '1px solid var(--border-hairline)',
                      color: 'var(--text-muted)',
                    }}
                  />
                </div>
              </div>

              {/* Organization */}
              <div>
                <label className="hud-label block mb-2">Organization / Department</label>
                <div className="relative">
                  <Building className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] pointer-events-none" />
                  <input
                    type="text"
                    value={organization}
                    onChange={(e) => setOrganization(e.target.value)}
                    placeholder="Department or Centre name"
                    className="w-full pl-10 pr-4 py-2.5 rounded text-sm outline-none transition-all"
                    style={{
                      background: 'var(--surface-2)',
                      border: '1px solid var(--border-hairline)',
                      color: 'var(--text-primary)',
                    }}
                  />
                </div>
              </div>

              <div className="flex items-center justify-between pt-2">
                <span className="text-xs text-[var(--text-faint)]">
                  Changes persist locally to your active session.
                </span>
                <button
                  type="submit"
                  className="btn-primary flex items-center gap-2 cursor-pointer"
                  style={{ padding: '8px 18px', fontSize: '0.78rem' }}
                >
                  <Save className="w-3.5 h-3.5" />
                  {profileSaved ? 'Saved!' : 'Save Profile'}
                </button>
              </div>
            </form>
          </div>

          {/* Password Change Card */}
          <div
            className="p-6 rounded-lg border border-[var(--border-hairline)] relative overflow-hidden"
            style={{
              background: 'var(--surface-1)',
              boxShadow: '0 4px 20px -2px rgba(0,0,0,0.3)',
            }}
          >
            <div className="flex items-center gap-3 mb-6 pb-4 border-b border-[var(--border-hairline)]">
              <div
                className="w-10 h-10 rounded-md flex items-center justify-center"
                style={{
                  background: 'color-mix(in srgb, var(--amber) 14%, transparent)',
                  border: '1px solid color-mix(in srgb, var(--amber) 30%, transparent)',
                }}
              >
                <KeyRound className="w-5 h-5 text-[var(--amber)]" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-[var(--text-primary)]">
                  Change Clearance Password
                </h2>
                <p className="text-xs text-[var(--text-muted)]">
                  Update your access password for this ground control terminal.
                </p>
              </div>
            </div>

            <form onSubmit={handleChangePassword} className="flex flex-col gap-4">
              {/* Current Password */}
              <div>
                <label className="hud-label block mb-2">Current Password</label>
                <div className="relative">
                  <Lock className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] pointer-events-none" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    placeholder="Enter current password"
                    className="w-full pl-10 pr-10 py-2.5 rounded text-sm outline-none transition-all font-mono"
                    style={{
                      background: 'var(--surface-2)',
                      border: '1px solid var(--border-hairline)',
                      color: 'var(--text-primary)',
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-faint)] hover:text-[var(--text-primary)] transition-colors p-1"
                  >
                    {showPassword ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              {/* New Password & Confirm Password */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="hud-label block mb-2">New Password</label>
                  <div className="relative">
                    <Lock className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] pointer-events-none" />
                    <input
                      type={showPassword ? 'text' : 'password'}
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      placeholder="Min. 8 characters"
                      className="w-full pl-10 pr-4 py-2.5 rounded text-sm outline-none transition-all font-mono"
                      style={{
                        background: 'var(--surface-2)',
                        border: '1px solid var(--border-hairline)',
                        color: 'var(--text-primary)',
                      }}
                    />
                  </div>
                </div>

                <div>
                  <label className="hud-label block mb-2">Confirm New Password</label>
                  <div className="relative">
                    <Lock className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-faint)] pointer-events-none" />
                    <input
                      type={showPassword ? 'text' : 'password'}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="Re-enter new password"
                      className="w-full pl-10 pr-4 py-2.5 rounded text-sm outline-none transition-all font-mono"
                      style={{
                        background: 'var(--surface-2)',
                        border: '1px solid var(--border-hairline)',
                        color: 'var(--text-primary)',
                      }}
                    />
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-end pt-2">
                <button
                  type="submit"
                  disabled={savingPassword}
                  className="btn-primary cursor-pointer"
                  style={{
                    padding: '8px 18px',
                    fontSize: '0.78rem',
                    background: 'linear-gradient(135deg, var(--amber) 0%, #D97706 100%)',
                  }}
                >
                  {savingPassword ? 'Updating Password…' : 'Update Password'}
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Right Column: Clearance Telemetry & Actions */}
        <div className="flex flex-col gap-6">
          {/* Clearance Card */}
          <div
            className="p-6 rounded-lg border border-[var(--border-hairline)] flex flex-col gap-4"
            style={{ background: 'var(--surface-1)' }}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-10 h-10 rounded-md flex items-center justify-center"
                style={{
                  background: 'color-mix(in srgb, var(--green) 14%, transparent)',
                  border: '1px solid color-mix(in srgb, var(--green) 30%, transparent)',
                }}
              >
                <Shield className="w-5 h-5 text-[var(--green)]" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                  Clearance &amp; Role
                </h3>
                <span className="hud-label text-[0.62rem] text-[var(--cyan)]">
                  {user?.role || 'ISRO Specialist'}
                </span>
              </div>
            </div>

            <div className="divide-y divide-[var(--border-subtle)] text-xs font-mono">
              <div className="py-2.5 flex items-center justify-between">
                <span className="text-[var(--text-muted)]">Terminal ID</span>
                <span className="text-[var(--text-primary)]">TERM-GEO-09</span>
              </div>
              <div className="py-2.5 flex items-center justify-between">
                <span className="text-[var(--text-muted)]">Station Node</span>
                <span className="text-[var(--cyan)]">ISRO-GEO-VQA-01</span>
              </div>
              <div className="py-2.5 flex items-center justify-between">
                <span className="text-[var(--text-muted)]">Downlink Uplink</span>
                <span className="text-[var(--text-primary)]">2.2 GHz (S-Band)</span>
              </div>
              <div className="py-2.5 flex items-center justify-between">
                <span className="text-[var(--text-muted)]">Security Protocol</span>
                <span className="text-[var(--green)]">TLS 1.3 / AES-GCM</span>
              </div>
            </div>
          </div>

          {/* Quick Action: Proceed to Dashboard */}
          <div
            className="p-6 rounded-lg border border-[var(--cyan)]/30 flex flex-col gap-3 relative overflow-hidden"
            style={{
              background: 'linear-gradient(135deg, color-mix(in srgb, var(--cyan) 10%, var(--surface-1)) 0%, var(--surface-1) 100%)',
            }}
          >
            <span className="hud-label text-[var(--cyan)]">Ready for Operations?</span>
            <h4 className="text-sm font-semibold text-[var(--text-primary)]">
              Proceed to Mission Analysis Dashboard
            </h4>
            <p className="text-xs text-[var(--text-muted)] leading-relaxed">
              Launch Single Image, Bi-Temporal, or Optical+SAR sensor workflows.
            </p>
            <Link
              href="/"
              className="btn-primary w-full flex items-center justify-center gap-2 mt-2"
              style={{ padding: '10px 16px', fontSize: '0.78rem' }}
            >
              <span>Go to Dashboard</span>
              <ArrowRight className="w-4 h-4 arrow-slide" />
            </Link>
          </div>

          {/* Logout Section */}
          <div
            className="p-6 rounded-lg border border-[var(--border-hairline)] flex flex-col gap-3"
            style={{ background: 'var(--surface-1)' }}
          >
            <h4 className="text-sm font-semibold text-[var(--text-primary)] flex items-center gap-2">
              <LogOut className="w-4 h-4 text-[var(--red)]" />
              Session Control
            </h4>
            <p className="text-xs text-[var(--text-muted)] leading-relaxed">
              End your active ground-control session and return to the secure login gateway.
            </p>
            <button
              type="button"
              onClick={logout}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded text-xs font-semibold text-[var(--red)] border border-[var(--red)]/30 bg-[var(--red)]/10 hover:bg-[var(--red)]/20 transition-all cursor-pointer mt-1"
            >
              <LogOut className="w-3.5 h-3.5" />
              Sign Out / Terminate Session
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
